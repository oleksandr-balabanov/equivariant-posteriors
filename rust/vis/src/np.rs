use std::path::Path;

fn to_array_d<T>(data: Vec<T>, shape: Vec<u64>, order: npyz::Order) -> ndarray::ArrayD<T> {
    use ndarray::ShapeBuilder;

    let shape = shape.into_iter().map(|x| x as usize).collect::<Vec<_>>();
    let true_shape = shape.set_f(order == npyz::Order::Fortran);

    ndarray::ArrayD::from_shape_vec(true_shape, data)
        .unwrap_or_else(|e| panic!("shape error: {}", e))
}

pub fn load_npy(path: &Path) -> std::io::Result<ndarray::ArrayD<f32>> {
    let bytes = std::fs::read(path)?;
    let reader = npyz::NpyFile::new(&bytes[..])?;
    let shape = reader.shape().to_vec();
    let order = reader.order();
    let data = reader.into_vec::<f32>()?;

    let nda = to_array_d(data.clone(), shape.clone(), order);
    Ok((nda))
}
