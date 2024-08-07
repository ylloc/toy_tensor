# toy_tensor
Toy library for multidimensional arrays

```rust
fn main() {
    let mut tensor = Tensor::new([10, 20, 30], 3); // shape = (10, 20, 30)
    let mut tensor_view = tensor.as_unique(); // unique (mutable) view
    tensor_view[[1, 1, 1]] = 333;
    assert!(
        tensor[[1, 1, 1]] == 333,
        "because `tensor_view` is view above `tensor` data"
    );
    let mut tensor_reshaped = tensor_view.reshape([1, 600]); // reshape to (1, 600)
    let owned = tensor_reshaped.to_owned(); // creates a copy of `tensor`
}
```
