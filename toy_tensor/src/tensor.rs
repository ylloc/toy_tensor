#![allow(dead_code)]
use std::ops::{Index, IndexMut};

#[derive(Clone, Debug, Default)]
pub struct InnerBuffer<T: Copy + Clone> {
    buffer: Vec<T>,
    size: usize,
}

pub struct Tensor<T: Copy + Clone, const DIM: usize> {
    buffer: InnerBuffer<T>,
    strides: [usize; DIM],
    shape: [usize; DIM],
}

impl<T: Copy + Clone, const DIM: usize> Tensor<T, DIM> {
    pub fn new(shape: [usize; DIM], val: T) -> Self {
        let size = shape.iter().product::<usize>();
        let mut strides = [1; DIM];
        for i in (0..DIM - 1).rev() {
            strides[i] = strides[i + 1] * shape[i];
        }
        Tensor {
            buffer: InnerBuffer {
                buffer: vec![val; size],
                size,
            },
            strides,
            shape,
        }
    }

    pub fn as_shared(&self) -> TensorView<T, DIM> {
        TensorView {
            reference: &self.buffer,
            strides: self.strides,
            shape: self.shape,
        }
    }

    pub fn as_unique(&mut self) -> MutTensorView<T, DIM> {
        MutTensorView {
            reference: &mut self.buffer,
            strides: self.strides,
            shape: self.shape,
        }
    }
}

#[derive(Clone)]
pub struct TensorView<'a, T: Copy + Clone, const DIM: usize> {
    reference: &'a InnerBuffer<T>,
    strides: [usize; DIM],
    shape: [usize; DIM],
}

impl<'a, T: Copy + Clone, const DIM: usize> TensorView<'a, T, DIM> {
    pub fn to_owned(&self) -> Tensor<T, DIM> {
        Tensor {
            buffer: self.reference.clone(),
            strides: self.strides,
            shape: self.shape,
        }
    }
}

fn calculate_position<const DIM: usize>(strides: &[usize; DIM], index: &[usize; DIM]) -> usize {
    let mut pos = 0usize;
    for i in 0..DIM {
        pos += strides[DIM - i - 1] * index[i];
    }
    return pos;
}

impl<'a, T: Copy + Clone, const DIM: usize> Index<[usize; DIM]> for TensorView<'a, T, DIM> {
    type Output = T;

    fn index(&self, index: [usize; DIM]) -> &Self::Output {
        let pos = calculate_position(&self.strides, &index);
        self.reference.buffer.get(pos).unwrap()
    }
}

pub struct MutTensorView<'a, T: Copy + Clone, const DIM: usize> {
    reference: &'a mut InnerBuffer<T>,
    strides: [usize; DIM],
    shape: [usize; DIM],
}

impl<'a, T: Copy + Clone, const DIM: usize> MutTensorView<'a, T, DIM> {
    pub fn to_owned(&mut self) -> Tensor<T, DIM> {
        Tensor {
            buffer: self.reference.clone(),
            strides: self.strides,
            shape: self.shape,
        }
    }
}

impl<'a, T: Copy + Clone, const DIM: usize> Index<[usize; DIM]> for MutTensorView<'a, T, DIM> {
    type Output = T;

    fn index(&self, index: [usize; DIM]) -> &Self::Output {
        let pos = calculate_position(&self.strides, &index);
        self.reference.buffer.get(pos).unwrap()
    }
}

impl<'a, T: Copy + Clone, const DIM: usize> IndexMut<[usize; DIM]> for MutTensorView<'a, T, DIM> {
    fn index_mut(&mut self, index: [usize; DIM]) -> &mut Self::Output {
        let pos = calculate_position(&self.strides, &index);
        self.reference.buffer.get_mut(pos).unwrap()
    }
}

impl<T: Copy + Clone, const DIM: usize> Index<[usize; DIM]> for Tensor<T, DIM> {
    type Output = T;

    fn index(&self, index: [usize; DIM]) -> &Self::Output {
        let pos = calculate_position(&self.strides, &index);
        self.buffer.buffer.get(pos).unwrap()
    }
}

impl<T: Copy + Clone, const DIM: usize> IndexMut<[usize; DIM]> for Tensor<T, DIM> {
    fn index_mut(&mut self, index: [usize; DIM]) -> &mut Self::Output {
        let pos = calculate_position(&self.strides, &index);
        self.buffer.buffer.get_mut(pos).unwrap()
    }
}

impl<'a, T: Copy + Clone, const DIM: usize> TensorView<'a, T, DIM> {
    pub fn reshape<const NDIM: usize>(self, new_shape: [usize; NDIM]) -> TensorView<'a, T, NDIM> {
        assert!(new_shape.iter().product::<usize>() == self.shape.iter().product::<usize>());

        let mut new_strides = [1; NDIM];
        for i in (0..NDIM - 1).rev() {
            new_strides[i] = new_strides[i + 1] * new_shape[i];
        }
        return TensorView {
            reference: self.reference,
            shape: new_shape,
            strides: new_strides,
        };
    }
}

impl<'a, T: Copy + Clone, const DIM: usize> MutTensorView<'a, T, DIM> {
    pub fn reshape<const NDIM: usize>(
        self,
        new_shape: [usize; NDIM],
    ) -> MutTensorView<'a, T, NDIM> {
        assert!(new_shape.iter().product::<usize>() == self.shape.iter().product::<usize>());

        let mut new_strides = [1; NDIM];
        for i in (0..NDIM - 1).rev() {
            new_strides[i] = new_strides[i + 1] * new_shape[i];
        }
        return MutTensorView {
            reference: self.reference,
            strides: new_strides,
            shape: new_shape,
        };
    }
}