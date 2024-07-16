mod tensor;

#[cfg(test)]
mod tests {
    use tensor::Tensor;

    use super::*;

    #[test]
    fn test() {
        let mut tensor = Tensor::new([10, 20, 30], 3);
        let tensor_view = tensor.as_unique();
        let mut tensor_reshaped = tensor_view.reshape([200, 30]).to_owned();
        tensor_reshaped[[1, 1]] = 2;
    }
}
