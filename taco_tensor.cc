#include <string>
#include <list>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "taco/tensor.h"
#include "taco/format.h"

using namespace tensorflow;

REGISTER_OP("TacoTensor")
    .Input("dimensions: int32")
    .Input("format: string")
    .Input("values: double")
    .Input("indices: N*int32")
    .Attr("N: int")
    .Output("taco_tensor: double")
    .Output("taco_index: string")
    ;

using namespace tensorflow;

class TacoTensorOp : public OpKernel {
 public:
  explicit TacoTensorOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
   //gathering tensor values
    const Tensor& format_string_tensor = context->input(1);
    auto format_string_input = format_string_tensor.flat<string>();
    bool alldense=true;

    //creating TACO format
    int FS = format_string_input.size();

    const Tensor& dim_tensor = context->input(0);
    auto dim_flat = dim_tensor.flat<int32>();
    int32* dim_ptr=(int32*)dim_flat.data();
    std::vector<int32> dim_input (dim_ptr,dim_ptr+FS);

    std::vector<taco::ModeType> format_input(FS);
    for (int i=0; i<FS; i++) {
      if (format_string_input(i)=="dense") {
        format_input[i] = taco::Dense;
      } else if (format_string_input(i)=="sparse") {
        format_input[i] = taco::Sparse;
        alldense=false;
      }								//else raise error
    }
    taco::Format user_format(format_input);

    //creating TACO tensor with proper dimensions and typing
    taco::Tensor<double> tensor("default", dim_input, user_format);
    OpInputList indices_tensor;
    OP_REQUIRES_OK(context, context->input_list("indices", &indices_tensor));
    const Tensor& values_tensor = context->input(2);
    auto values = values_tensor.flat<double>();

    int i=0;
    if (alldense) {
      std::vector<int32> index (FS,0);
      bool same=false;
      while(!same){
        tensor.insert(index, values(i));
        i++;
        index[FS-1]++;
        bool diff=false;
        for (int j=(FS-1);j>0; j--) {
          if (index[j]==dim_input[j]) {
            index[j-1]++;
            index[j]=0;
          } else {diff=true;}
        }
        if ((diff==false)&&(index[0]==dim_input[0])) {
          same=true;
        }
      }
    } else {
      for(auto& index_tensor:indices_tensor){
        int32* idx=(int32*)index_tensor.flat<int32>().data();
        std::vector<int32> indices (idx, idx+FS);
        tensor.insert(indices, values(i));
        i++;
      }
    }
    tensor.pack();
    							//Technically this makes a duplicate of all the data so I can delete all previous tensors but idk how
    
    //creating output matrix in proper data format
						    	//This makes a second copy of the data that I should also be able to delete...
    const std::initializer_list<int64> N = {(long long int) tensor.getStorage().getValues().getSize()}; //this will be changed too later
    Tensor* taco_tensor = NULL;
    Tensor* taco_index_tensor = NULL;
    const std::initializer_list<int64> _N = {(long long int)FS+2};
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape(_N), &taco_index_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(N), &taco_tensor));
    auto taco_index = taco_index_tensor->flat<string>();
    auto taco_val = taco_tensor->flat<double>();
    const double* data = static_cast<const double*>(tensor.getStorage().getValues().getData());
    std::copy_n(data, tensor.getStorage().getValues().getSize(), taco_val.data());
    std::ostringstream stream;
    for (size_t i =0; i<FS; i++) {
      for (size_t j=0; j<tensor.getStorage().getIndex().getModeIndex(i).numIndexArrays();j++) {
        taco::storage::Array arr = tensor.getStorage().getIndex().getModeIndex(i).getIndexArray(j);
        const int32* data = static_cast<const int32*>(arr.getData());
        if (arr.getSize() > 0) {
          stream << data[0];
        }
        for (size_t i = 1; i < arr.getSize(); i++) {
          stream << "," << data[i];
        }
        if (j==0){stream << "|";}
      }
      stream << " ";
    }
    string dim_text;
    for (auto&dim : dim_input) { dim_text=dim_text+std::to_string(dim)+" ";}
    taco_index(0) = dim_text;
    int k=1;
    for (int i=0;i<FS;i++) {taco_index(k)=format_string_input(i);k++;}
    taco_index(k) = stream.str();
  }
};

REGISTER_KERNEL_BUILDER(Name("TacoTensor").Device(DEVICE_CPU), TacoTensorOp);

/*
#define REGISTER_KERNEL(param) \
  REGISTER_KERNEL_BUILDER(Name("TacoTensor").Device(DEVICE_CPU).TypeConstraint<param>("type"), TacoTensorOp<param>);

REGISTER_KERNEL(int32);
REGISTER_KERNEL(std::list(int));
//REGISTER_KERNEL(float);
//REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
*/
