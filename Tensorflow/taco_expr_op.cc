#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "taco/tensor.h"
#include "taco/parser/parser.h"
#include "taco/format.h"
#include "taco/util/timers.h"
#include "taco/ir/ir.h"

using namespace tensorflow;

REGISTER_OP("TacoExprOp")
    .Input("expression: string")
    .Input("taco_tensors: N * double")
    .Input("taco_indices: N * string")
    .Attr("which_grad: int = -1")
    .Output("taco_tensor: T * double")
    .Output("taco_index: T * string")
    .Output("taco_exp: string")
    .Output("ordering: int32")
    .Attr("N: int")
    .Attr("T: int = 2")
    ;

using namespace tensorflow;

class TacoExprOp : public OpKernel {
 public:
  explicit TacoExprOp(OpKernelConstruction* context) : OpKernel(context) {
   OP_REQUIRES_OK(context, context->GetAttr("which_grad",&which_grad));
  }

  taco::Tensor<double> * convert(Tensor ind_tensor, Tensor val_tensor) {
    /**
      converts Tensorflow tensors containing taco index and value metadata
      into a taco tensor as a pointer
    */
    //taco::util::Timer convertTimer;
    //convertTimer.start();


    auto ind_flat = ind_tensor.flat<string>();

    string dim_text = ind_flat(0);
    std::vector<int> dim_vec;
    std::istringstream iss_dim(dim_text);
    string s;
    while (iss_dim>>s) {dim_vec.push_back(std::stoi(s));}

    std::vector<taco::ModeType> format_input(ind_flat.size()-2);
    for (int i=1; i<ind_flat.size()-1; i++) {
      if (ind_flat(i)=="dense") {
        format_input[i-1] = taco::Dense;
      } else if (ind_flat(i)=="sparse") {
        format_input[i-1] = taco::Sparse;
      } //else {std::cout<<"Invalid ModeType"<<std::endl;}
    }
    taco::Format format(format_input);
 
    string index_text = ind_flat(ind_flat.size()-1);
    std::istringstream iss(index_text);
    std::vector<taco::storage::ModeIndex> index_vec;
    for(string index_text;iss>>s;){
      std::vector<int32> * data = new std::vector<int32>();
      size_t split_idx = s.find("|");
      string s1 = s.substr(0,split_idx);
      string s2 = s.substr(split_idx+1);
      std::stringstream ss(s1);
      string temp;
      while (std::getline(ss,temp,',')) {data->push_back(std::stoi(temp));}
      std::vector<taco::storage::Array> arr_vec;
      if (s2.size()>0){
        std::vector<int32> * data2 = new std::vector<int32>();
        std::stringstream ss2(s2);
        while(std::getline(ss2,temp,',')){data2->push_back(std::stoi(temp));}
        int32 * arr_data = &data->at(0);
        //is this a safe conversion?
        int32 * arr_data2 = &data2->at(0);
        taco::storage::Array arr1(taco::Int(32), arr_data, data->size());
        taco::storage::Array arr2(taco::Int(32), arr_data2, data2->size());
        arr_vec.push_back(arr1);
        arr_vec.push_back(arr2);
      } else {
        int32 * arr_data = &data->at(0);
        taco::storage::Array arr(taco::Int(32), arr_data,data->size());
        arr_vec.push_back(arr);
      }
      taco::storage::ModeIndex dim_idx(arr_vec);
      index_vec.push_back(dim_idx);
    }
    taco::storage::Index index(format, index_vec);

    const auto& values_flat = val_tensor.flat<double>();
    const int S = values_flat.size();
    double * copy_values = new double[S];
    for(int i=0;i<S;i++){ copy_values[i]=values_flat(i);} 
    taco::storage::Array values(taco::Float(64), copy_values, S);

    taco::Tensor<double> *taco_ptr = new taco::Tensor<double>(dim_vec, format);
    taco_ptr->getStorage().setValues(values);
    taco_ptr->getStorage().setIndex(index);

    //convertTimer.stop();
    //std::cout<<"convertTimer:"<<convertTimer.getResult()<<std::endl;
    return taco_ptr;
  }

  void getIndexString(taco::Tensor<double> & result, Tensor * index_tensor) {
    /**
      converts taco Tensor indices into string metadata that is then 
      placed into a Tensorflow string tensor
    */
    //taco::util::Timer gISTimer;
    //gISTimer.start(); 
    std::ostringstream stream;
    auto index = index_tensor->flat<string>();
    for (size_t i =0; i<result.getOrder(); i++) {
      for (size_t j=0; j<result.getStorage().getIndex().getModeIndex(i).numIndexArrays();j++) {
        taco::storage::Array arr = result.getStorage().getIndex().getModeIndex(i).getIndexArray(j);
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

    std::ostringstream index_os;
    index_os << result.getStorage().getIndex();
    string index_str = index_os.str();

    string dim_text;
    for (auto&dim : result.getDimensions()) { dim_text=dim_text+std::to_string(dim)+" ";}

    index(0) = dim_text;
    int j=1;
    for (auto&format : result.getFormat().getModeTypes()) {
      if(format==taco::Dense){
        index(j)="dense";
      }else{
        index(j)="sparse";
      }
      j++;
    }
    index(j) = stream.str();
    //gISTimer.stop();
    //std::cout<<"\tgIS Timer:"<<gISTimer.getResult()<<std::endl;
  }

  void Compute(OpKernelContext* context) override {
    /**
      evaluates expression using TACO and computes
      gradient 
    */

    //taco::util::Timer wholeTimer;
    //wholeTimer.start();
    //Gather input data
    const Tensor& expr_tensor = context->input(0);
    auto expr_flat = expr_tensor.flat<string>();
    OpInputList indices_tensor;
    OP_REQUIRES_OK(context, context->input_list("taco_indices", &indices_tensor));
    OpInputList values_tensor;									//check if same size else throw error
    OP_REQUIRES_OK(context, context->input_list("taco_tensors", &values_tensor));

    std::vector<taco::Tensor<double>*> tensor_list;
    for(int i =0; i<indices_tensor.size();i++){tensor_list.push_back(convert(indices_tensor[i], values_tensor[i]));} 



    //std::cout<<"starting parseTimer"<<std::endl;
    //taco::util::Timer parseTimer;
    //parseTimer.start();
    //Replace variables in expression with correct TACO default names
    string temp = expr_flat(0);
    size_t begin = temp.find("=")+1;
    size_t ignore_idx = temp.substr(0,begin).find('(');						//if find goes beyond eol then throw error
    string new_expr = "result"+temp.substr(ignore_idx,begin-ignore_idx);

    string expr = temp.substr(begin);
    std::vector<string> rename;
    std::map<string,taco::IndexVar> index_vars;
    string new_expr_space = new_expr;
    string namestring;
    string namestring_space;

    for(int i=0;i<expr.length();i++) {
      char ch=expr[i];
      if(!(ch==' ')){
        if((ch=='+')|(ch=='-')|(ch=='*')|(ch=='/')){
          if(!namestring.empty()){
            ignore_idx = namestring.find_last_of("(");
            std::stringstream ss_index(namestring.substr(ignore_idx+1, namestring.size()-2));
            while (std::getline(ss_index,temp,',')) {
              if(index_vars.count(temp)==0) {
                taco::IndexVar ivar(temp);
                index_vars[temp]=ivar;
              }
            }
            auto rename_idx=std::find(rename.begin(), rename.end(),namestring.substr(0,ignore_idx));
            if(!(rename_idx == rename.end())){
              string name = tensor_list[rename_idx-rename.begin()]->getName();
              new_expr+=name+namestring.substr(ignore_idx);
              new_expr_space=new_expr_space+" "+name+" "+namestring_space.substr(namestring_space.find_last_of("("));
            }else{
              rename.push_back(namestring.substr(0,ignore_idx));
              string name = tensor_list[rename.size()-1]->getName();
              new_expr+=name+namestring.substr(ignore_idx);
              new_expr_space=new_expr_space+" "+name+" "+namestring_space.substr(namestring_space.find_last_of("("));
            }
            new_expr+=ch;
            new_expr_space=new_expr_space+" "+ch;
            namestring.clear();
            namestring_space.clear();
          }
        }else{
          if(namestring.empty() && ch=='(') {new_expr+=ch; new_expr_space+=" ";new_expr_space+=ch;} else {
            namestring+=ch;
            namestring_space= namestring_space+" "+ch;
          }
        }
      }
    }
    if(!namestring.empty()){
      size_t ignore_idx = namestring.find("(");
      auto rename_idx=std::find(rename.begin(), rename.end(),namestring.substr(0,ignore_idx));
      if(!(rename_idx == rename.end())){
        string name = tensor_list[rename_idx-rename.begin()]->getName();
        new_expr+=name+namestring.substr(ignore_idx);
        new_expr_space=new_expr_space+" "+name+" "+namestring_space.substr(namestring_space.find_last_of("("));
     }else{
        rename.push_back(namestring.substr(0,ignore_idx));
        string name = tensor_list[rename.size()-1]->getName();
        new_expr+=name+namestring.substr(ignore_idx);
        new_expr_space=new_expr_space+" "+name+" "+namestring_space.substr(namestring_space.find_last_of("("));
      }
    }




    //std::cout<<"starting gradient parser"<<std::endl;
    string grad_expr;
    string new_expr_paren;
    int unique_count=0;

    //Parse expression to create new expression to calculate the gradient of the Tensor indexed by which_grad
    if(which_grad >-1){
      size_t which_grad_start_idx = new_expr_space.find("A")+1;
      unique_count = std::stoi(new_expr_space.substr(which_grad_start_idx,new_expr_space.find("(")-which_grad_start_idx));
      size_t start_point = new_expr_space.find("A"+std::to_string(which_grad+unique_count));

      string first_half = new_expr_space.substr(0, start_point);
      string second_half = new_expr_space.substr(start_point);
      size_t first_half_counter =first_half.length()-1;
      size_t second_half_counter = 0;
      std::vector<size_t> first_splits;
      std::vector<size_t> second_splits;							//seriously just reverse the gdd string

      for(size_t brac=0; brac<first_half.length();brac++) {
        if (first_half[brac]=='('){
          first_splits.push_back(brac);
        } else if (first_half[brac]==')') {
          first_splits.pop_back();
        }
      }
      for(size_t brac=second_half.length();brac>0;brac--) {
        size_t brac_idx = brac -1;
        if (second_half[brac_idx]==')'){
          second_splits.push_back(brac_idx);
        } else if (second_half[brac_idx]=='(') {
          second_splits.pop_back();
        }
      }

											     	 //these can be made into functions since theyre the same...
      string fin_first_half;
      string fin_second_half;
      size_t last_index = first_half.length() -1;
      first_splits.push_back(last_index);
      second_splits.push_back(0);

      for (int idx=first_splits.size()-1;idx>=0;idx--){
        first_half_counter = first_splits[idx];
        size_t limit=0;
        if(idx!=0) {limit=first_splits[idx-1];}
        while ((first_half_counter>limit) && (first_half[first_half_counter] != '+')) {
          if(first_half[first_half_counter] == '*') {
            if (first_half[first_half_counter-2] == ')') {
              if(first_half[first_half_counter-4] == ')') {
                int level =1;
                size_t count = 3;
                while(level!=0) {
                  count++;
                  if (first_half[first_half_counter-count]=='(') {level--;} else if (first_half[first_half_counter-count]==')') {level++;}
                } 
                fin_first_half=first_half.substr(first_half_counter-count, count+1)+fin_first_half;
                first_half_counter-=count;
              } else {
                size_t end_mul = first_half.substr(0,first_half_counter-2).find_last_of("A");	//2 is highly dependent on format 
                fin_first_half=first_half.substr(end_mul, first_half_counter-end_mul+1)+fin_first_half;
                first_half_counter=end_mul;
              }
            }  
          }else{first_half_counter--;}
        }
        first_half_counter--;
      }
    
      for (int idx=(second_splits.size()-1); idx>=0;idx--){
        size_t limit=second_half.length()-1;
        if (idx>0){limit=second_splits[idx-1];}
        while ((second_half_counter < limit) && (second_half[second_half_counter] != '+')) {
          if(second_half[second_half_counter] == '*') {
            if (second_half[second_half_counter+2] == 'A') {
              size_t end_mul = second_half.substr(second_half_counter+2).find(")")+second_half_counter+3; //2 or space, 1 for length
              fin_second_half+=second_half.substr(second_half_counter,end_mul-second_half_counter);
              second_half_counter=end_mul;
            } else if(second_half[second_half_counter+2] == '(') {
              int level =1;
              size_t count = 2;
              while(level!=0) {
                count++;
                if (second_half[second_half_counter+count]=='(') {level++;} else if (second_half[second_half_counter+count]==')') {level--;}
              }
              count++;
              fin_second_half+=second_half.substr(second_half_counter, count); 
              second_half_counter+=count;
            }  
          }else{second_half_counter++;}
        }  
        second_half_counter++; 
      }

      if ((fin_first_half[fin_first_half.length()-1] != ')')&&(fin_second_half[0]=='*')) {
        fin_second_half=fin_second_half.substr(1);
      } else if ((fin_first_half[fin_first_half.length()-1] == '*')&&((fin_second_half[0]!='(')||(fin_second_half[0]!='A'))) {
        fin_first_half = fin_first_half.substr(0, fin_first_half.length()-1);
      }
      new_expr_paren = fin_first_half + fin_second_half;
    }



    //std::cout<<"starting variable replacement"<<std::endl;
    string replacement_var; 

    //Replace indexVars and remove the Tensor indexed by which_grad from the gradient expression
    if(which_grad>-1){
      int A_grad_expr = 0;
      int A = 0;
      int ch =0;
      std::stringstream ss_grad(new_expr_paren);
      string temp;
      bool skip = false;
      bool matched = false;
      int skip_num = 0;
      while (std::getline(ss_grad,temp,' ')) {
        ch = ch + 1 + temp.length();
        if (temp == 'A' + std::to_string(which_grad+unique_count)) {
          int walk = tensor_list[which_grad]->getOrder();
          if( A > 0){
            int x=A; 
            int y=ch; 
            string temp2;
            string temp1;
            string index_change;
            string ss_small_1_string = new_expr_paren.substr( x, 4*tensor_list[which_grad-1]->getOrder());
            std::stringstream ss_small_1(new_expr_paren.substr( x, 4*tensor_list[which_grad-1]->getOrder()));	//highly dep on format 
            std::stringstream ss_small_2(new_expr_paren.substr(y, 4*walk)); 					//highly dep on format
            while (std::getline(ss_small_2,temp2,' ')) {
              y= y+1+temp2.length();
              x=A;
              if (!matched) {index_change.clear();} else {index_change += temp2;} 
              while(!matched && std::getline(ss_small_1, temp1,' ')){
                x = x+1+temp1.length();
                if(temp1 == temp2 && temp1 != "(" && temp1 != ","){
                  matched =true;
                  index_change = index_change.substr(0,index_change.length()-1);
                } else {index_change += temp1;}
              }
              if(!matched){ 
                ss_small_1.str(std::string());
                ss_small_1.clear();
                ss_small_1 << ss_small_1_string;
              }
            }
            if (matched) {grad_expr = grad_expr.substr(0, A_grad_expr); grad_expr+=index_change; grad_expr+=")";}
          } 
          skip = true;
          if (matched) {skip_num = 2 * tensor_list[which_grad]->getOrder() + 1;} else {skip_num = 2 * tensor_list[which_grad]->getOrder() + 2;}
        } else if (temp[0] == 'A') { A_grad_expr = grad_expr.length()+temp.length(); A = ch;}
        if(!skip) {grad_expr +=temp;} else{ if(skip_num!=0){skip_num--;} else {skip=false;}}
      }
    }






    bool ones_matrix = false;
    int transpose_point = -1;
    string return_expr = "\0";
    int ordering = -1;
    string replacement_vars;
    //std::cout<<"creating return expression"<<std::endl;

    //Calculates return expression to calculate the differential
    if((which_grad>-1) &&(grad_expr.length()!=0)) {
      size_t start = new_expr.find("(")+1;
      string result_var = new_expr.substr(start, new_expr.find(")")-start); 
      string matrix_var = new_expr.substr(new_expr.find("A"+std::to_string(which_grad+unique_count)));
      return_expr += matrix_var.substr(0,matrix_var.find(")")+1);
      return_expr += '=';
      start = matrix_var.find("(")+1;
      matrix_var = matrix_var.substr(start, matrix_var.find(")")-start);
      string transposed_vars;
      string new_result_var;
      string new_matrix_var;
      int matrix_var_count=0;
      int result_var_count=0;
      std::vector<bool> matrix_var_list (matrix_var.length());

      for(auto ch:result_var){
        bool same=false;
        if ((ch!=',')){
          for(size_t i=0;i<matrix_var.length();i++) {
            if (ch==matrix_var[i]) {same=true;matrix_var_list[i]=true;}
          }
          if(!same){
            new_result_var+=ch;
            result_var_count++;
          }
        }
      }
      for(size_t i=0;i<matrix_var.length();i++) {if((matrix_var[i]!=',')&&(!matrix_var_list[i])){matrix_var_count++;new_matrix_var+=matrix_var[i];}}

      if (result_var[0]==matrix_var[0]) {
        if (result_var==matrix_var){
          ordering = 2;
          start = new_expr.find("(")+1;
          replacement_vars = new_expr.substr(start, new_expr.find(")")-start);
          transposed_vars = replacement_vars;
          transpose_point = 0;
        }else {
          replacement_vars = new_matrix_var + new_result_var;
          ordering = 2;
          transposed_vars = new_result_var + new_matrix_var;
          transpose_point = result_var_count;
        }
      }else {
        replacement_vars = new_result_var + new_matrix_var;
        ordering = 1;
        transpose_point = matrix_var_count;
        transposed_vars = new_matrix_var + new_result_var;
      } 

      
      string replacement_var_string;
      string transpose_var_string;
      for(auto&ch:transposed_vars) {transpose_var_string=transpose_var_string+ch+',';}
      for (auto& ch:replacement_vars) {replacement_var_string = replacement_var_string +ch+',';} 
      transpose_var_string.pop_back();
      replacement_var_string = replacement_var_string.substr(0,replacement_var_string.length()-1);
      grad_expr = "result("+replacement_var_string+")="+grad_expr;
      return_expr +="transpose("+transpose_var_string+")*"+new_expr.substr(0,new_expr.find(")")+1);
    } else {
      ones_matrix=true;
    }

    //parseTimer.stop();
    //std::cout<<"parseTimer:"<<parseTimer.getResult()<<std::endl;
    //taco::util::Timer tacoTimer;
    //tacoTimer.start();
    //taco::util::Timer prepareParseTimer;
    //prepareParseTimer.start();

    //calculates result using TACO
    std::map<string, taco::Format> formats;
    std::map<string, std::vector<int>> tensor_sizes;
    std::map<string, taco::TensorBase> loaded_tensors;
    string name_it;
    for(auto&tensor_it:tensor_list){
      name_it=tensor_it->getName(); 
      formats[name_it]=tensor_it->getFormat();
      tensor_sizes[name_it]=tensor_it->getDimensions();
      loaded_tensors[name_it]=*tensor_it;
    }
    //prepareParseTimer.stop();
    //std::cout<<"\tprepareParseTimer:"<<prepareParseTimer.getResult()<<std::endl;
    									//What happens when we have (i(k)*j(k))-b(k) I get incompatible dimensions??? 
    //taco::util::Timer parseResultTimer;
    //parseResultTimer.start();
    taco::parser::Parser pars(new_expr_space, formats, tensor_sizes, loaded_tensors, 42);
    pars.parse();
    taco::Tensor<double> result = pars.getResultTensor();
    //parseResultTimer.stop();
    //std::cout<<"\t\tparseResultTimer:"<<parseResultTimer.getResult()<<std::endl;

    //error message if failures from line 441 of https://github.com/tensor-compiler/taco/blob/8123c991673ec5e8916b52bc9fd94ede1853b081/tools/taco.cpp
    //std::cout<<"going to get result of "<<new_expr_space<<std::endl;

    //taco::util::Timer compileTimer;
    //compileTimer.start();
    //mutex
    if (funcs.empty()) {
      statements = result.get_lowers();
      funcs = result.functionize(statements);
      //if(compute_func == nullptr) {std::cout<<"still empty why"<<std::endl;} else {std::cout<<"not empty"<<std::endl;}
      result.apply_funcs(funcs, &statements[0], &statements[1]);
    } else {
      //std::cout<<"parallelism THO"<<std::endl;
      result.apply_funcs(funcs, &statements[0], &statements[1]);
      //std::cout<<compute_func_string<<std::endl;
    }
    //result.compile();
    //Tensor tensor = context->mutable_input(0, true);
    
    //compileTimer.stop();
    //std::cout<<"\tcompileTimer:"<<compileTimer.getResult()<<std::endl;
    //taco::util::Timer assembleTimer;
    //assembleTimer.start();
    result.assemble();
    //assembleTimer.stop();
    //std::cout<<"\tassembleTimer:"<<assembleTimer.getResult()<<std::endl;
    //taco::util::Timer computeTimer;
    //computeTimer.start();
    result.compute();
    //computeTimer.stop();
    //std::cout<<"\tcomputeTimer:"<<computeTimer.getResult()<<std::endl;
    //std::cout<<"which is\n"<<result<<std::endl;
    //Preparing Tensorflow ouputs
    //taco::util::Timer outputTimer;
    //outputTimer.start();
    OpOutputList idx_outputs;
    OpOutputList val_outputs;
    Tensor * taco_exp = nullptr;
    Tensor * tordering = nullptr;
    OP_REQUIRES_OK(context, context->output_list("taco_index",&idx_outputs));
    OP_REQUIRES_OK(context,context->output_list("taco_tensor",&val_outputs));
    OP_REQUIRES_OK(context, context->allocate_output("taco_exp",TensorShape({1}),&taco_exp));
    OP_REQUIRES_OK(context, context->allocate_output("ordering",TensorShape({}), &tordering));
    auto taco_exp_flat = taco_exp->flat<string>();
    auto ordering_flat = tordering->flat<int32>();
    taco_exp_flat(0) = return_expr;    
    ordering_flat(0) = ordering;
    const int FS = result.getFormat().getOrder();
    const std::initializer_list<int64> N = {(long long int) result.getStorage().getValues().getSize()}; //this will be changed too later
    Tensor* taco_tensor = nullptr;
    Tensor* taco_index_tensor = nullptr;
    Tensor* grad_tensor = nullptr;
    Tensor* grad_index_tensor = nullptr;
    //outputTimer.stop();
    //std::cout<<"outputTimer:"<<outputTimer.getResult()<<std::endl;



    //Calculates gradient using TACO
    if(!ones_matrix) {
      //taco::util::Timer gradTimer;
      //gradTimer.start();
      //taco::util::Timer gradParserTimer;
      //gradParserTimer.start();
      std::vector<taco::ModeType> sparse_format (replacement_vars.length(), taco::Sparse);
      //std::vector<taco::ModeType> sparse_format (grad_result.getOrder(), taco::Sparse);
      formats["result"] = sparse_format;
      taco::parser::Parser grad_pars(grad_expr, formats, tensor_sizes, loaded_tensors, 42);
      grad_pars.parse();
      //gradParserTimer.stop();
      //std::cout<<"\t\tgradParserTimer:"<<gradParserTimer.getResult()<<std::endl;
      taco::Tensor<double> grad_result = grad_pars.getResultTensor();
      //std::cout<<"calculating gradient of "<<grad_expr<<std::endl;
      //this could be optimized but it won't be in tacotensor lol

      //grad_result.setFormat(sparse_format);
      //grad_result.pack();

      //taco::Tensor<double> grad_result(placeholder_result.getDimensions(), sparse_format);

      
      //grad_result.compile();
      if (grad_funcs.empty()) {
        grad_statements = grad_result.get_lowers();
        grad_funcs = grad_result.functionize(grad_statements);
      //if(compute_func == nullptr) {std::cout<<"still empty why"<<std::endl;} else {std::cout<<"not empty"<<std::endl;}
        grad_result.apply_funcs(grad_funcs, &grad_statements[0], &grad_statements[1]);
      } else {
        grad_result.apply_funcs(grad_funcs, &grad_statements[0], &grad_statements[1]);
      //std::cout<<compute_func_string<<std::endl;
      }
      grad_result.assemble();
      grad_result.compute();
      //gradTimer.stop();
      //std::cout<<"\tgradTimer:"<<gradTimer.getResult()<<std::endl;
      //tacoTimer.stop();
      //std::cout<<"returns\n"<<grad_result<<std::endl;
      //std::cout<<"tacoTimer:"<<tacoTimer.getResult()<<std::endl;
      //taco::util::Timer transposeTimer;
      //transposeTimer.start();
      taco::storage::Index grad_og_idx = grad_result.getStorage().getIndex();
      taco::Format grad_format = grad_result.getFormat();

      std::vector<int> grad_dim = grad_result.getDimensions();
      int order = grad_dim.size();
      std::vector<int> grad_transpose_dim (order);
      std::vector<taco::ModeType> grad_transpose_mode (order);
      for(int i =0; i<(order);i++){
        grad_transpose_dim[(transpose_point+i)%(order)]=grad_dim[i];
        auto test = grad_format.getModeTypes();
        grad_transpose_mode[(transpose_point+i)%(order)]=grad_format.getModeTypes()[i];
      }
      taco::Format grad_transpose_format(grad_transpose_mode);
      taco::Tensor<double> grad_transpose(grad_transpose_dim, grad_transpose_format);    

      //Recovers coordinates and tranposes them
      std::vector<int32> vertex (order,0);
      									//would be easier if i could use coordinatebuffer
      									//order level, array, start idx, indx limit is given,
      std::vector<int32> limit (order,0);
      std::vector<int32> count (order,-1);
      count[order-1] = 0;

      while(limit[order-1]<grad_result.getStorage().getValues().getSize()) {
        for(int i=order-1; i>=0;i--) {
          if(count[i]==limit[i]) {
            if (i>0){
              count[i-1] = count[i-1]+1;
              if(grad_format.getModeTypes()[i-1]==taco::Dense){
                vertex[(i-1+transpose_point)%(order)] = (count[i-1])%grad_dim[i-1];
              } else {
                vertex[(i-1+transpose_point)%(order)] = ((int32 *) grad_og_idx.getModeIndex(i-1).getIndexArray(1).getData())[count[i-1]];
              }
            }
            if(grad_format.getModeTypes()[i]==taco::Dense){
              limit[i] = limit[i] + grad_dim[i];
            } else {
              limit[i] = ((int32 *) grad_og_idx.getModeIndex(i).getIndexArray(0).getData())[count[i-1]+1];
            }
         }
        }
        while(count[order-1]<limit[order-1]){
          if(grad_format.getModeTypes()[order-1]==taco::Dense){
            vertex[(order-1+transpose_point)%(order)] = (count[order-1])%grad_dim[order-1];

          } else {
            vertex[(order-1+transpose_point)%(order)] = ((int32 *) (grad_og_idx.getModeIndex(order-1).getIndexArray(1).getData()))[count[order-1]];
          }
          grad_transpose.insert(vertex, ((const double*)grad_result.getStorage().getValues().getData())[count[order-1]]);
          count[order-1] = count[order-1] +1;
        }
      }
      //taco::util::Timer transposePackTimer;
      //transposePackTimer.start();
      grad_transpose.pack();
      //transposePackTimer.stop();
      //std::cout<<"transposePackTimer:"<<transposePackTimer.getResult()<<std::endl;
      //transposeTimer.stop();
      //std::cout<<"transposeTimer:"<<transposeTimer.getResult()<<std::endl;
      //taco::util::Timer endTimer;
      //endTimer.start();
      OP_REQUIRES_OK(context, idx_outputs.allocate(1, TensorShape({(long long int) (2+grad_transpose.getOrder())}), &grad_index_tensor));
      OP_REQUIRES_OK(context, val_outputs.allocate(1, TensorShape({(long long int) grad_transpose.getStorage().getValues().getSize()}), &grad_tensor));
      OP_REQUIRES_OK(context, idx_outputs.allocate(0,TensorShape({(long long int) (2+result.getOrder())}), &taco_index_tensor));
      OP_REQUIRES_OK(context, val_outputs.allocate(0, TensorShape({(long long int) result.getStorage().getValues().getSize()}), &taco_tensor));
      getIndexString(result, taco_index_tensor);
      getIndexString(grad_transpose, grad_index_tensor);
      auto grad_val = grad_tensor->flat<double>();
      auto taco_val = taco_tensor->flat<double>();
      const double* data = static_cast<const double*>(result.getStorage().getValues().getData());
      std::copy_n(data, result.getStorage().getValues().getSize(), taco_val.data());
      const double* data_grad = static_cast<const double*>(grad_transpose.getStorage().getValues().getData());
      std::copy_n(data_grad, grad_transpose.getStorage().getValues().getSize(), grad_val.data());
      //endTimer.stop();
      //std::cout<<"endTimer:"<<endTimer.getResult()<<std::endl;
    } else {
      //tacoTimer.stop();
      //std::cout<<"no gradient tacoTimer:"<<tacoTimer.getResult()<<std::endl;
      //taco::util::Timer endTimer;
      //endTimer.start();
      OP_REQUIRES_OK(context, idx_outputs.allocate(0, TensorShape({2+FS}), &taco_index_tensor));
      OP_REQUIRES_OK(context, val_outputs.allocate(0, TensorShape(N), &taco_tensor));
      OP_REQUIRES_OK(context, idx_outputs.allocate(1, TensorShape({1}), &grad_index_tensor));
      OP_REQUIRES_OK(context, val_outputs.allocate(1, TensorShape({1}), &grad_tensor));
      
      auto taco_val = taco_tensor->flat<double>();
      const double* data = static_cast<const double*>(result.getStorage().getValues().getData());
      std::copy_n(data, result.getStorage().getValues().getSize(), taco_val.data());
      getIndexString(result, taco_index_tensor);

      auto grad_index = grad_index_tensor->flat<string>();
      grad_index(0) = "\0";
      auto grad_val = grad_tensor->flat<double>();
      grad_val(0) = -1;
      //endTimer.stop();
      //std::cout<<"endTimer:"<<endTimer.getResult()<<std::endl;
    }

    for (auto& tensor: tensor_list){delete tensor;}
    //wholeTimer.stop();
    //std::cout<<"wholeTimer:"<<wholeTimer.getResult()<<std::endl;
  }

  private:
     int which_grad;
     std::string funcs;
     std::string grad_funcs;
     std::vector<taco::ir::Stmt> statements;
     std::vector<taco::ir::Stmt> grad_statements;
     //taco::ir::Stmt assemble_func;
     //taco::ir::Stmt compute_func;
     //taco::ir::Stmt grad_assemble_func;
     //taco::ir::Stmt grad_compute_func;
};

REGISTER_KERNEL_BUILDER(Name("TacoExprOp").Device(DEVICE_CPU), TacoExprOp);
