use std::{collections::HashMap, f32::EPSILON};

use anyhow::anyhow;
use glam::{Mat4, Quat, Vec3};
use openusd::{
    sdf::{self, AbstractData, Path},
    usda,
    usdc::{self},
};

use crate::{camera::Camera, material::Material, mesh::Mesh, vertex::EngineVertex};

pub struct Parser {
    pub meshes: Vec<Mesh>,
    pub cameras: Vec<Camera>,
    pub lights: Vec<Mesh>,

    material_map: HashMap<String, Material>,
}

impl Parser {
    pub fn new() -> Self {
        Parser {
            meshes: Vec::new(),
            cameras: Vec::new(),
            lights: Vec::new(),
            material_map: HashMap::new(),
        }
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

        self.meshes.iter_mut().for_each(|mesh| {
            if let Some(path) = &mesh.material_path {
                if self.material_map.contains_key(path) {
                    mesh.material = Some((*self.material_map.get(path).unwrap()).clone());
                }
            }
        });

        self.lights.iter_mut().for_each(|mesh| {
            if let Some(path) = &mesh.material_path {
                if self.material_map.contains_key(path) {
                    mesh.material = Some((*self.material_map.get(path).unwrap()).clone());
                }
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
                    Ok(xform) => {
                        if Parser::is_affine_like(xform, 1e-4) {
                            new_xform = curr_xform * xform;
                        } else {
                            println!("SKIPPED XFORM FOR {}", path);
                        }
                    }
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
                let mesh = self.parse_mesh(path_obj, data, curr_xform)?;
                self.meshes.push(mesh);
            }
            Some("Light") => {
                let light = self.parse_mesh(path_obj, data, curr_xform)?;
                self.lights.push(light);
            }
            Some("Camera") => {
                let camera = self.parse_camera(path, data, curr_xform)?;
                self.cameras.push(camera);
            }
            Some("Material") => {
                let material = self.parse_material(path, data)?;
                self.material_map.insert(path.to_string(), material);
            }
            None => return Err(anyhow!("No type name found.")),
            _ => {
                // println!("{} {}", path, type_name.unwrap());
            }
        }

        if !data.has_field(path_obj, "primChildren") {
            return Ok(false);
        }

        let prim_children = data
            .get(path_obj, "primChildren")?
            .into_owned()
            .try_as_token_vec()
            .unwrap();

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

    fn is_affine_like(m: Mat4, eps: f32) -> bool {
        // translation w should be ~1
        let w_ok = (m.w_axis.w - 1.0).abs() < eps;

        // all entries finite
        let finite = m.to_cols_array().iter().all(|x| x.is_finite());

        w_ok && finite
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
            .expect("FAILED TO UNWRAP TRANSLATE");
        let translation = Vec3::new(
            translation_vec[0] as f32,
            translation_vec[1] as f32,
            translation_vec[2] as f32,
        );

        let rotation_obj = data.get(rotate_path, "default")?;
        let rotation_vec = rotation_obj
            .try_as_vec_3f_ref()
            .expect("FAILED TO UNWRAP ROTATE");
        let rot = Vec3::new(
            rotation_vec[0].to_radians() as f32,
            rotation_vec[1].to_radians() as f32,
            rotation_vec[2].to_radians() as f32,
        );

        // USD's rotateXYZ uses extrinsic rotations (around fixed parent axes).
        // For extrinsic XYZ, use intrinsic ZYX with reversed angle order.
        let rotation = Quat::from_euler(glam::EulerRot::ZYX, rot[2], rot[1], rot[0]);

        let scale_obj = data.get(scale_path, "default")?;
        let scale_vec = scale_obj
            .try_as_vec_3f_ref()
            .expect("FAILED TO UNWRAP SCALE");
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
        path: &Path,
        data: &mut Box<dyn AbstractData>,
        curr_xform: glam::Mat4,
    ) -> Result<Mesh, anyhow::Error> {
        // TODO: SUPPORT NON TRIANGULAR MESHES
        let face_vertex_count_path = path.append_property("faceVertexCounts")?;
        let face_vertex_counts: Vec<i32> = data
            .get(&face_vertex_count_path, "default")?
            .into_owned()
            .try_as_int_vec()
            .expect("Could not find faceVertexCounts");

        if face_vertex_counts.iter().any(|count| *count != 3) {
            return Err(anyhow!("Non Triangular Mesh Found. {}", path));
        }

        let indices_path = path.append_property("faceVertexIndices")?;
        let indices = data
            .get(&indices_path, "default")?
            .into_owned()
            .try_as_int_vec()
            .expect("Could not find faceVertexIndices")
            .iter()
            .map(|i| *i as u32)
            .collect();

        let points_path = path.append_property("points")?;
        let points_data: Vec<f32> = data
            .get(&points_path, "default")?
            .into_owned()
            .try_as_vec_3f_ref()
            .expect("Could not find points")
            .to_vec();
        let vertices: Vec<&[f32]> = points_data.chunks(3).collect();

        let normals_path = path.append_property("normals")?;
        let normals_data: Vec<f32> = data
            .get(&normals_path, "default")?
            .into_owned()
            .try_as_vec_3f_ref()
            .expect("Could not find normals")
            .to_vec();
        let normals: Vec<&[f32]> = normals_data.chunks(3).collect();

        let engine_vertices = (0..vertices.len())
            .map(|i| {
                let point = vertices[i];
                let normal = normals[i];

                EngineVertex {
                    position: [point[0], point[1], point[2]],
                    normal: [normal[0], normal[1], normal[2]],
                }
            })
            .collect();

        let material_list_path = path.append_property("material:binding")?;

        let mut material_path: Option<String> = None;
        if let Some(material_list_option) = data
            .get(&material_list_path, "targetPaths")?
            .into_owned()
            .try_as_path_list_op()
        {
            if material_list_option.explicit_items.len() > 0 {
                material_path = Some(material_list_option.explicit_items[0].to_string());
            }
        }

        Ok(Mesh {
            vertices: engine_vertices,
            indices,
            xform: curr_xform,
            name: path.to_string(),
            material_path,
            ..Default::default()
        })
    }

    fn parse_camera(
        &mut self,
        path: &str,
        data: &mut Box<dyn AbstractData>,
        curr_xform: glam::Mat4,
    ) -> Result<Camera, anyhow::Error> {
        let focal_length_path = &sdf::path(&format!("{}.focalLength", path))?;
        let focal_length = data
            .get(&focal_length_path, "default")?
            .into_owned()
            .try_as_float()
            .expect("Could not find focalLength");

        let horizontal_aperture_path = &sdf::path(&format!("{}.horizontalAperture", path))?;
        let horizontal_aperture = data
            .get(&horizontal_aperture_path, "default")?
            .into_owned()
            .try_as_float()
            .expect("Could not find horizontalAperture");

        let vertical_aperture_path = &sdf::path(&format!("{}.verticalAperture", path))?;
        let vertical_aperture = data
            .get(&vertical_aperture_path, "default")?
            .into_owned()
            .try_as_float()
            .expect("Could not find verticalAperture");

        Ok(Camera {
            focal_len: focal_length,
            sensor_x: horizontal_aperture,
            sensor_y: vertical_aperture,
            width: 1920,
            height: 1080,
            sample_count: 1,
            xform: curr_xform,
        })
    }

    fn parse_material(
        &mut self,
        path: &str,
        data: &mut Box<dyn AbstractData>,
    ) -> Result<Material, anyhow::Error> {
        let surface_path = &sdf::path(&format!("{}.outputs:surface", path))?;

        let connections = data
            .get(&surface_path, "connectionPaths")?
            .into_owned()
            .try_as_path_list_op();

        if connections.is_none() {
            return Err(anyhow!("Material had no connections for {}", path));
        }

        let connection_paths = connections.unwrap();
        if connection_paths.explicit_items.len() <= 0 {
            return Err(anyhow!("Connection paths empty for {}", path));
        }

        let primary_connection_path = &connection_paths.explicit_items[0].prim_path();
        let albedo_path = primary_connection_path.append_property("inputs:diffuseColor")?;
        let emission_path = primary_connection_path.append_property("inputs:emissiveColor")?;

        let albedo = data
            .get(&albedo_path, "default")?
            .into_owned()
            .try_as_vec_3f()
            .unwrap();

        let emission = vec_to_vec3_op(
            data.get(&emission_path, "default")?
                .into_owned()
                .try_as_vec_3f(),
        );

        Ok(Material {
            albedo: Vec3::new(albedo[0], albedo[1], albedo[2]),
            emission,
            name: path.to_string(),
        })
    }
}

fn vec_to_vec3_op(value: Option<Vec<f32>>) -> Option<Vec3> {
    match value {
        None => None,
        Some(vec) => Some(Vec3::new(vec[0], vec[1], vec[2])),
    }
}
