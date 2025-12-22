use std::{any::Any, fs};

use anyhow::{Error, anyhow};
use glam::{Mat4, Quat, Vec3};
use openusd::{
    sdf::{self, AbstractData},
    usdc::{self, CrateFile},
};

use crate::mesh::Mesh;

pub struct Parser {
    pub meshes: Vec<Mesh>,
}

impl Parser {
    pub fn new() -> Self {
        Parser { meshes: Vec::new() }
    }

    pub fn parse(&mut self, path: &String) -> Result<bool, anyhow::Error> {
        let mut data = usdc::read_file(path).unwrap();
        let root = data
            .get(&sdf::Path::abs_root(), "primChildren")?
            .into_owned()
            .try_as_token_vec()
            .unwrap();
        root.iter().for_each(|child| {
            let val = self.recursive_parse(
                format!("/{}", child).as_str(),
                &mut data,
                glam::Mat4::IDENTITY,
            );
            match val {
                Err(x) => println!("{:?}", x),
                _ => {}
            }
        });

        Ok(true)
    }

    fn recursive_parse(
        &mut self,
        path: &str,
        data: &mut Box<dyn AbstractData>,
        curr_xform: glam::Mat4,
    ) -> Result<bool, anyhow::Error> {
        let mut new_xform = curr_xform;
        let path_obj = &sdf::path(&path)?;
        if let Some(spec_type) = data.spec_type(path_obj) {
            if spec_type.to_string() != "Prim" {
                return Err(anyhow!("Not prim!"));
            }
        } else {
            return Err(anyhow!("Not prim!"));
        }

        let mut type_name: Option<String> = None;
        if data.has_field(path_obj, "typeName") {
            let type_obj = data.get(path_obj, "typeName")?;
            let token_ref = type_obj.into_owned().try_as_token().unwrap();
            type_name = Some(token_ref);
        }

        if type_name.is_none() {
            return Err(anyhow!("No typename found for path: {}", path));
        }

        match type_name.as_deref() {
            Some("Xform") => {
                let parsed_xform = self.parse_xform(path, data);
                match parsed_xform {
                    Ok(xform) => new_xform = curr_xform * xform,
                    _ => {}
                }
            }
            Some("Mesh") => {
                // let properties: Vec<String> = data
                //     .get(&path_obj, "properties")?
                //     .into_owned()
                //     .try_as_token_vec()
                //     .unwrap();
                // dbg!(properties);
                let mesh = self.parse_mesh(path, data, curr_xform)?;
                self.meshes.push(mesh);
            }
            None => return Err(anyhow!("No type name found.")),
            _ => {}
        }

        let prim_children = data
            .get(path_obj, "primChildren")?
            .into_owned()
            .try_as_token_vec()
            .unwrap();

        // println!("New prims: {:?}", prim_children);

        let mut error = None;
        prim_children.iter().for_each(|child| {
            let val = self.recursive_parse(format!("{}/{}", path, child).as_str(), data, new_xform);
            if val.is_err() {
                error = Some(val)
            }
        });

        if !error.is_none() {
            return error.unwrap();
        }

        Ok(true)
    }

    fn parse_xform(
        &mut self,
        path: &str,
        data: &mut Box<dyn AbstractData>,
    ) -> Result<glam::Mat4, anyhow::Error> {
        let translate_path = &sdf::path(&format!("{}.xformOp:translate", path))?;
        let rotate_path = &sdf::path(&format!("{}.xformOp:rotateXYZ", path))?;
        let scale_path = &sdf::path(&format!("{}.xformOp:scale", path))?;
        if !data.has_field(translate_path, "default") {
            return Err(anyhow!("Translate not found for path: {}", path));
        }
        if !data.has_field(rotate_path, "default") {
            return Err(anyhow!("Rotate not found for path: {}", path));
        }
        if !data.has_field(scale_path, "default") {
            return Err(anyhow!("Scale not found for path: {}", path));
        }

        let translation_obj = data.get(translate_path, "default")?;
        let translation_vec = translation_obj
            .try_as_vec_3d_ref()
            .expect("FAILED TO UNWRAP");
        let translation = Vec3::new(
            translation_vec[0] as f32,
            translation_vec[1] as f32,
            translation_vec[2] as f32,
        );

        let rotation_obj = data.get(translate_path, "default")?;
        let rotation_vec = rotation_obj.try_as_vec_3d_ref().expect("FAILED TO UNWRAP");
        let rotation = Quat::from_euler(
            glam::EulerRot::XYZ,
            rotation_vec[0] as f32,
            rotation_vec[1] as f32,
            rotation_vec[2] as f32,
        );

        let scale_obj = data.get(translate_path, "default")?;
        let scale_vec = scale_obj.try_as_vec_3d_ref().expect("FAILED TO UNWRAP");
        let scale: Vec3 = Vec3::new(
            scale_vec[0] as f32,
            scale_vec[1] as f32,
            scale_vec[2] as f32,
        );

        let xform: Mat4 = glam::Mat4::from_scale_rotation_translation(scale, rotation, translation);

        Ok(xform)
    }

    fn parse_mesh(
        &mut self,
        path: &str,
        data: &mut Box<dyn AbstractData>,
        curr_xform: glam::Mat4,
    ) -> Result<Mesh, anyhow::Error> {
        // TODO: SUPPORT NON TRIANGULAR MESHES
        let face_vertex_count_path = &sdf::path(&format!("{}.faceVertexCounts", path))?;
        let face_vertex_counts: Vec<i32> = data
            .get(&face_vertex_count_path, "default")?
            .into_owned()
            .try_as_int_vec()
            .expect("Could not find faceVertexCounts");

        if face_vertex_counts.iter().any(|count| *count != 3) {
            return Err(anyhow!("Non Triangular Mesh Found."));
        }

        let indices_path = &sdf::path(&format!("{}.faceVertexIndices", path))?;
        let indices = data
            .get(&indices_path, "default")?
            .into_owned()
            .try_as_int_vec()
            .expect("Could not find faceVertexIndices");

        let points_path = &sdf::path(&format!("{}.points", path))?;
        let vertices = data
            .get(&points_path, "default")?
            .into_owned()
            .try_as_vec_3f_ref()
            .expect("Could not find points")
            .to_vec()
            .chunks(3)
            .map(|chunk| glam::vec3(chunk[0], chunk[1], chunk[2]))
            .collect();

        println!("GOT HERE");

        Ok(Mesh {
            vertices,
            indices,
            xform: curr_xform,
            name: path.to_string(),
        })
    }
}
