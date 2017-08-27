# TacoTF
MIT COMMIT TACO &amp; Tensorflow integration. 
Description: TACO is a compiler for fast sparse matrix calculations (see this paper,) and Tensorflow is a graph-representation language for machine learning. TacoTF integrates TACO and Tensorflow by creating TACO nodes that can be used in Tensorflow graphs.

Usage:

1. Build Tensorflow from source (see https://www.tensorflow.org/install/install_sources) with Bazel
2. Download the TACO source (see https://github.com/tensor-compiler/taco)
3. Replace the tensor.cpp and tensor.h with the corresponding files from this repo.
2. Build TACO (see https://github.com/tensor-compiler/taco)
3. Navigate to the tensorflow/tensorflow/WORKSPACE directory and add this code snippet:
        new_local_repository(
        name = "taco",
        path = "/path/to/dir/taco",
        build_file = "taco.BUILD",
        )
3. Move all the non-Python files from this repo into /tensorflow/tensorflow/core/user_ops
4. Build each TACO op by running bazel build --config opt //tensorflow/core/user_ops:name_of_op.so
5. Edit the Python op module directory paths if necessary and run the Python script to run TACO with Tensorflow

Documentation:

[to be added]
