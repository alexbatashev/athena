#ifndef POLAR_GRAPH_OPS
#define POLAR_GRAPH_OPS

include "PolarGraphDialect.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def NodeOp : PolarGraph_Op<"node", [AutomaticAllocationScope,
                                    FunctionLike, 
                                    IsolatedFromAbove, Symbol]> {
  let summary = "Polar Graph Node";
  // todo extend description with samples when dialect is formed.
  let description = [{
    Defines a sequence of operations required to complete node execution. 
    This includes allocation of tensors, locking tensors for particular access 
    mode, performing operations on tensors.

    The operation has one region that corresponds to the body of node function.
  }];

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<"OpBuilder& builder, OperationState &result, StringRef name, "
              "FunctionType type, size_t nodeId, size_t clusterId, "
              "ArrayRef<NamedAttribute> attrs = {}">
  ];

  let extraClassDeclaration = [{
    friend class OpTrait::FunctionLike<NodeOp>;
    unsigned getNumFuncArguments() { return getType().getNumInputs(); }
    unsigned getNumFuncResults() { return getType().getNumResults(); }
    static StringRef getNodeIdAttrName() {
      return "node_id";
    }
    static StringRef getClusterIdAttrName() {
      return "cluster_id";
    }
  }];
  // let printer = [{ printNodeOp(p, *this); }];
  // let parser = [{ return parseNodeOp(parser, result); }];
}

def GraphOp : PolarGraph_Op<"graph", 
                          [AutomaticAllocationScope, FunctionLike, 
                          IsolatedFromAbove, Symbol,
                          SingleBlockImplicitTerminator<"GraphTerminatorOp">]> {
  let summary = "Polar Graph operation";
  // todo extend description with samples when dialect is formed.
  let description = [{
  TBD
  }];

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<"OpBuilder& builder, OperationState &result, StringRef name, "
              "ArrayRef<NamedAttribute> attrs = {}">
  ];

  let extraClassDeclaration = [{
    friend class OpTrait::FunctionLike<GraphOp>;
    unsigned getNumFuncArguments() { return getType().getNumInputs(); }
    unsigned getNumFuncResults() { return getType().getNumResults(); }
  }];
}

def ReturnOp : PolarGraph_Op<"return", [NoSideEffect, HasParent<"NodeOp">,
                                        Terminator, ReturnLike]> {
  let summary = "return operation";
  let description = [{
    Returns result of Athena Graph Node computation.
  }];

  // fixme should this be AnyTensor?
  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [OpBuilder<
    "OpBuilder& b, OperationState &result", [{ build(b, result, llvm::None); }]
  >];
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def GraphTerminatorOp : PolarGraph_Op<"graph_terminator", 
                                    [Terminator, HasParent<"GraphOp">, 
                                    NoSideEffect]> {
  let summary = "graph terminator operation";
  let description = [{
    Indicates end of graph computation.
  }];
}

def EvalOp : PolarGraph_Op<"eval", [CallOpInterface]> {
  let summary = "Evaluate graph node";
  let description = [{ TBD }];

  let arguments = (ins FlatSymbolRefAttr:$node, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [OpBuilder<
      "OpBuilder& builder, OperationState &result, NodeOp node,"
      "ValueRange operands = {}", [{
        result.addOperands(operands);
        result.addAttribute("node", builder.getSymbolRefAttr(node));
        result.addTypes(node.getType().getResults());
  }]>];

  let extraClassDeclaration = [{
      /// Get the argument operands to the called function.
      operand_range getArgOperands() {
        return {arg_operand_begin(), arg_operand_end()};
      }

      operand_iterator arg_operand_begin() { return operand_begin(); }
      operand_iterator arg_operand_end() { return operand_end(); }

      /// Return the callee of this operation.
      CallInterfaceCallable getCallableForCallee() {
        return getAttrOfType<SymbolRefAttr>("node");
      }
  }];

  let assemblyFormat = [{
    $node `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def SliceOp : PolarGraph_Op<"slice", [NoSideEffect]> {
  let summary = "Get sub-tensor by first dimension offset";
  let description = [{ TBD }];

  let arguments = (ins AnyType:$id, AnyTensor:$tensor);
  let results = (outs AnyTensor);

  let builders = [OpBuilder<"OpBuilder& builder, OperationState &result, "
                            "Value slice, Value tensor">];
  let skipDefaultBuilders = 1;
}

def CreateTensorOp : PolarGraph_Op<"create_tensor", [NoSideEffect]> {
  let summary = "Get tensor by virtual address";
  let description = [{ TBD }];

  let arguments = (ins IndexAttr:$virtual_address);
  let results = (outs AnyTensor);

  let builders = [OpBuilder<"OpBuilder& builder, OperationState &result, "
                            "uint64_t virtualAddress, RankedTensorType type">];
  let skipDefaultBuilders = 1;
  let extraClassDeclaration = [{
    static llvm::StringRef getVirtualAddressAttrName() {
      return "virtual_address";
    }
  }];
}

def AllocOp : PolarGraph_Op<"alloc"> {
  let summary = "Allocates memory for tensor";
  let description = [{}];

  let arguments = (ins AnyTensor:$target);
}

def LockOp : PolarGraph_Op<"lock"> {
  let summary = "Locks data on device";
  let description = [{
    Athena LLVM backend Allocators explicitly manage memory both on host and 
    devices. Memory can be moved from one domain to another. To prevent this, 
    tensors that are about to be accessed must be "locked" on a certain device.

    Nodes can lock tensors for read and read_write access. Multiple devices are
    allowed to lock tensor for read, and only one device is allowed to lock
    tensor for read_write.
  }];

  // todo change to enum.
  let arguments = (ins StrAttr:$lock_type, AnyTensor:$locked_tensor);
}

def ReleaseOp : PolarGraph_Op<"release"> {
  let summary = "Releases data";
  let description = [{
    Notifies Athena LLVM backend Allocator, that tensor is no longer in use and
    its memory can be freed.
  }];

  let arguments = (ins AnyTensor:$tensor);
}

def InvokeLoaderOp : PolarGraph_Op<"invoke_loader"> {
  let summary = "Invokes loader";
  let description = [{
    One way of getting data into tensor is invoking a loader routine. Users are
    allowed to write their own loaders. Loader must provide a C-like function
    that can be invoked on host.
  }];

  let arguments = (ins AnyTensor:$dest);

}

def BarrierOp : PolarGraph_Op<"barrier", [IsolatedFromAbove]> {
  let summary = "Blocks execution of the subsequent nodes";
  let description = [{ TBD }];

  let arguments = (ins IndexAttr:$clusterId);
}

//===----------------------------------------------------------------------===//
// PolarAI Graph builtins
//===----------------------------------------------------------------------===//

def AddOp : PolarGraph_Op<"add", [ComputationalOpInterface]> {
  let summary = "Element-wise addition operation";
  let description = [{ TBD }];

  let arguments = (ins AnyTensor:$a, AnyType:$scaleA, AnyTensor:$b,
                       AnyType:$scaleB, AnyTensor:$out);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<"OpBuilder& builder, OperationState& result, "
                            "Value a, Value scaleA, Value b, Value scaleB, "
                            "Value out", [{
    result.addOperands(a);
    result.addOperands(scaleA);
    result.addOperands(b);
    result.addOperands(scaleB);
    result.addOperands(out);
  }]>];
  let extraClassDeclaration = [{
    llvm::StringRef getKernelName() {
      return "fadd";
    }
    void produceKernel(OpBuilder&, Block::BlockArgListType);
  }];
}

def Conv2DOp : PolarGraph_Op<"conv2d", [ComputationalOpInterface]> {
  let summary = "2D convolution";
  let description = [{TBD}];

  let arguments = (ins AnyTensor:$input, AnyTensor:$conv_kernel, AnyTensor:$out);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<"OpBuilder& builder, OperationState& result, "
                            "Value input, Value kernel, Value out", [{
    result.addOperands({input, kernel, out});
  }]>];
  let extraClassDeclaration = [{
    llvm::StringRef getKernelName() {
      return "fconv2d";
    }
    void produceKernel(OpBuilder&, Block::BlockArgListType);
  }];
}

def CopyOp : PolarGraph_Op<"copy", [ComputationalOpInterface]> {
  let summary = "Element-wise copy operation";
  let description = [{ TBD }];

  let arguments = (ins AnyTensor:$input, AnyTensor:$out);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<"OpBuilder& builder, OperationState& result, "
                            "Value input, Value out", [{
    result.addOperands(input);
    result.addOperands(out);
  }]>];
  let extraClassDeclaration = [{
    llvm::StringRef getKernelName() {
      return "fcopy";
    }
    void produceKernel(OpBuilder&, Block::BlockArgListType);
  }];
}

def DivideOp : PolarGraph_Op<"divide", [ComputationalOpInterface]> {
  let summary = "Element-wise division operation";
  let description = [{ TBD }];

  let arguments = (ins AnyTensor:$numerator, AnyTensor:$denominator, 
                   AnyTensor:$out);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<"OpBuilder& builder, OperationState& result, "
                            "Value numerator, Value denominator, Value out", [{
    result.addOperands(numerator);
    result.addOperands(denominator);
    result.addOperands(out);
  }]>];
  let extraClassDeclaration = [{
    llvm::StringRef getKernelName() {
      return "fdivide";
    }
    void produceKernel(OpBuilder&, Block::BlockArgListType);
  }];
}

def LogLossOp : PolarGraph_Op<"logloss", [ComputationalOpInterface]> {
  let summary = "Element-wise logistic loss operation";
  let description = [{ TBD }];

  let arguments = (ins AnyTensor:$predicted, AnyTensor:$groundTruth, 
                   AnyTensor:$out);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<"OpBuilder& builder, OperationState& result, "
                            "Value predicted, Value groundTruth, Value out", [{
    result.addOperands(predicted);
    result.addOperands(groundTruth);
    result.addOperands(out);
  }]>];
  let extraClassDeclaration = [{
    llvm::StringRef getKernelName() {
      return "flogloss";
    }
    void produceKernel(OpBuilder&, Block::BlockArgListType) {}
  }];
}

def MatMulOp : PolarGraph_Op<"matmul", [ComputationalOpInterface]> {
  let summary = "Matrix-matrix multiplication operation";
  let description = [{ TBD }];

  let arguments = (ins AnyTensor:$left, AnyTensor:$right, AnyTensor:$out, 
                   BoolAttr:$transpose_left, BoolAttr:$transpose_right);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<"OpBuilder& builder, OperationState& result, "
                            "Value left, Value right, Value out, "
                            "bool transpLeft, bool transpRight", [{
    result.addOperands(left);
    result.addOperands(right);
    result.addOperands(out);
    auto transposeLeftAttr = builder.getBoolAttr(transpLeft);
    auto transposeRightAttr = builder.getBoolAttr(transpRight);
    result.addAttribute("transpose_left", transposeLeftAttr);
    result.addAttribute("transpose_right", transposeRightAttr);
  }]>];
  let extraClassDeclaration = [{
    llvm::StringRef getKernelName() {
      if (transpose_right() && transpose_left()) {
        return "fmatmul_t_t";
      } else if (transpose_right()) {
        return "fmatmul_f_t";
      } else if (transpose_left()) {
        return "fmatmul_t_f";
      } else {
        return "fmatmul_f_f";
      }
    }
    void produceKernel(OpBuilder&, Block::BlockArgListType);
  }];
}

def MulOp : PolarGraph_Op<"mul", [ComputationalOpInterface]> {
  let summary = "Element-wise multiplication operation";
  let description = [{ TBD }];

  let arguments = (ins AnyTensor:$a, AnyTensor:$b, AnyTensor:$out);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<"OpBuilder& builder, OperationState& result, "
                            "Value a, Value b, Value out", [{
    result.addOperands(a);
    result.addOperands(b);
    result.addOperands(out);
  }]>];
  let extraClassDeclaration = [{
    llvm::StringRef getKernelName() {
      return "fmul";
    }
    void produceKernel(OpBuilder&, Block::BlockArgListType);
  }];
}

def MulConcatOp : PolarGraph_Op<"mulconcat", [ComputationalOpInterface]> {
  let summary = "Element-wise multiplication operation";
  let description = [{ TBD }];

  let arguments = (ins AnyTensor:$gradient, AnyTensor:$localDerivative,
                   AnyTensor:$out);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<"OpBuilder& builder, OperationState& result, "
                            "Value gradient, Value localDerivative, Value out", 
                            [{
    result.addOperands(gradient);
    result.addOperands(localDerivative);
    result.addOperands(out);
  }]>];
  let extraClassDeclaration = [{
    llvm::StringRef getKernelName() {
      return "fmulconcat";
    }
    void produceKernel(OpBuilder&, Block::BlockArgListType);
  }];
}

def POOL_P_MAX : I64EnumAttrCase<"max", 0, "max">;

def PoolPredicateAttr : I64EnumAttr<"PoolPredicate", "", 
    [POOL_P_MAX]> {
  let cppNamespace = "::mlir::polar_graph";
}

def Pool2DOp : PolarGraph_Op<"pool2d", [ComputationalOpInterface]> {
  let summary = "2D pooling operation";
  let description = [{TBD}];

  let arguments = (ins 
    AnyTensor:$input, 
    AnyTensor:$out, 
    PoolPredicateAttr:$predicate,
    I64ArrayAttr:$window,
    I64ArrayAttr:$stride
  );
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<"OpBuilder& builder, OperationState& result, "
                            "Value input, Value out, PoolPredicate predicate, "
                            "ArrayRef<int64_t> sizes, ArrayRef<int64_t> stride", 
                            [{
    result.addOperands({input, out});
    result.addAttribute(
      getPredicateAttrName(), 
      builder.getI64IntegerAttr(static_cast<int64_t>(predicate)));
    result.addAttribute(getWindowAttrName(), builder.getI64ArrayAttr(sizes));
    result.addAttribute(getStrideAttrName(), builder.getI64ArrayAttr(stride));
  }]>];
  let extraClassDeclaration = [{
    static llvm::StringRef getPredicateAttrName() {
      return "predicate";
    }
    static llvm::StringRef getWindowAttrName() {
      return "window";
    }
    static llvm::StringRef getStrideAttrName() {
      return "stride";
    }
    llvm::StringRef getKernelName() {
      return "fpool";
    }
    void produceKernel(OpBuilder&, Block::BlockArgListType);
  }];

  let assemblyFormat = "$predicate $input `,` $out attr-dict `:` type($input) `->` type($out)";
}

def ReLUOp : PolarGraph_Op<"relu", [ComputationalOpInterface]> {
  let summary = "ReLU operation";
  let description = [{ TBD }];

  let arguments = (ins AnyTensor:$input, AnyTensor:$out);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<"OpBuilder& builder, OperationState& result, "
                            "Value input, Value out", [{
    result.addOperands({input, out});
  }]>];

  let extraClassDeclaration = [{
    llvm::StringRef getKernelName() {
      return "ftranspose";
    }
    void produceKernel(OpBuilder&, Block::BlockArgListType);
  }];
}

def SigmoidOp : PolarGraph_Op<"sigmoid", [ComputationalOpInterface]> {
  let summary = "Element-wise sigmoid operation";
  let description = [{ TBD }];

  let arguments = (ins AnyTensor:$input, AnyTensor:$out);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<"OpBuilder& builder, OperationState& result, "
                            "Value input, Value out", [{
    result.addOperands(input);
    result.addOperands(out);
  }]>];
  let extraClassDeclaration = [{
    llvm::StringRef getKernelName() {
      return "fsigmoid";
    }
    void produceKernel(OpBuilder&, Block::BlockArgListType);
  }];
}

def SoftmaxOp : PolarGraph_Op<"softmax", [ComputationalOpInterface]> {
  let summary = "Softmax operation";
  let description = [{ TBD }];

  let arguments = (ins AnyTensor:$input, AnyTensor:$out);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<"OpBuilder& builder, OperationState& result, "
                            "Value input, Value out", [{
    result.addOperands({input, out});
  }]>];

  let extraClassDeclaration = [{
    llvm::StringRef getKernelName() {
      return "ftranspose";
    }
    void produceKernel(OpBuilder&, Block::BlockArgListType);
  }];
}

def TransposeOp : PolarGraph_Op<"transpose", [ComputationalOpInterface]> {
  let summary = "Matrix transposition operation";
  let description = [{ TBD }];

  let hasCanonicalizer = 1;

  let arguments = (ins AnyTensor:$a, AnyTensor:$out);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<"OpBuilder& builder, OperationState& result, "
                            "Value a, Value out", [{
    result.addOperands(a);
    result.addOperands(out);
  }]>];

  let extraClassDeclaration = [{
    llvm::StringRef getKernelName() {
      return "ftranspose";
    }
    void produceKernel(OpBuilder&, Block::BlockArgListType) {}
  }];
}

def FillOp : PolarGraph_Op<"fill", [ComputationalOpInterface]> {
  let summary = "Fills tensor with value";
  let description = [{ TBD }];

  let arguments = (ins AnyType:$pattern, AnyTensor:$out);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<"OpBuilder& builder, OperationState& result, "
                            "Value pattern, Value out", [{
    result.addOperands(pattern);
    result.addOperands(out);
  }]>];
  let extraClassDeclaration = [{
    llvm::StringRef getKernelName() {
      return "ffill";
    }
    void produceKernel(OpBuilder&, Block::BlockArgListType) {}
  }];
}

#endif // ATHENA_GRAPH_OPS
