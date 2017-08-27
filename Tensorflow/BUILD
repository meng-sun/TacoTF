load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "zero_out.so",
    srcs = ["zero_out.cc"],
)

tf_custom_op_library(
    name = "tacorflow_base_op.so",
    srcs = ["tacorflow_base_op.cc"],
    deps = ["@taco//:libtaco"]
)

tf_custom_op_library(
    name = "taco_tensor.so",
    srcs = ["taco_tensor.cc"],
    deps = ["@taco//:libtaco"],
)

tf_custom_op_library(
    name = "taco_expr_op.so",
    srcs = ["taco_expr_op.cc"],
    deps = ["@taco//:libtaco"],
)

tf_custom_op_library(
    name = "taco_fast_dense.so",
    srcs = ["taco_fast_dense.cc"],
    deps = ["@taco//:libtaco"],
)
