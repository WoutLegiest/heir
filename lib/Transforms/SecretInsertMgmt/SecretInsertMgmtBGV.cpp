#include <utility>

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/MulDepthAnalysis/MulDepthAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"
#include "lib/Dialect/Mgmt/Transforms/Passes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Transforms/SecretInsertMgmt/Passes.h"
#include "lib/Transforms/SecretInsertMgmt/SecretInsertMgmtPatterns.h"
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"             // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_SECRETINSERTMGMTBGV
#include "lib/Transforms/SecretInsertMgmt/Passes.h.inc"

struct SecretInsertMgmtBGV
    : impl::SecretInsertMgmtBGVBase<SecretInsertMgmtBGV> {
  using SecretInsertMgmtBGVBase::SecretInsertMgmtBGVBase;

  void runOnOperation() override {
    // for Openfhe, use B/FV style mgmt: only relinearize, no level management.
    // still maintain the maximal level information though for lowering.
    if (moduleIsOpenfhe(getOperation())) {
      OpPassManager pipeline("builtin.module");
      pipeline.addPass(createSecretInsertMgmtBFV());
      (void)runPipeline(pipeline, getOperation());
      moduleSetBGV(getOperation());
      return;
    }

    // Helper for future lowerings that want to know what scheme was used
    moduleSetBGV(getOperation());

    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<SecretnessAnalysis>();
    solver.load<MulDepthAnalysis>();
    solver.load<LevelAnalysis>();

    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    RewritePatternSet patternsRelinearize(&getContext());
    patternsRelinearize.add<MultRelinearize<arith::MulIOp>>(
        &getContext(), getOperation(), &solver);
    (void)walkAndApplyPatterns(getOperation(), std::move(patternsRelinearize));

    RewritePatternSet patternsMultModReduce(&getContext());
    patternsMultModReduce.add<ModReduceBefore<arith::MulIOp>>(
        &getContext(), /*isMul*/ true, includeFirstMul, getOperation(),
        &solver);
    // tensor::ExtractOp = mulConst + rotate
    patternsMultModReduce.add<ModReduceBefore<tensor::ExtractOp>>(
        &getContext(), /*isMul*/ true, includeFirstMul, getOperation(),
        &solver);
    // isMul = true and includeFirstMul = false here
    // as before yield we want mulResult to be mod reduced
    patternsMultModReduce.add<ModReduceBefore<secret::YieldOp>>(
        &getContext(), /*isMul*/ true, /*includeFirstMul*/ false,
        getOperation(), &solver);
    (void)walkAndApplyPatterns(getOperation(),
                               std::move(patternsMultModReduce));

    // when other binary op operands level mismatch
    // includeFirstMul not used for these ops
    RewritePatternSet patternsAddModReduce(&getContext());
    patternsAddModReduce.add<ModReduceBefore<arith::AddIOp>>(
        &getContext(), /*isMul*/ false, /*includeFirstMul*/ false,
        getOperation(), &solver);
    patternsAddModReduce.add<ModReduceBefore<arith::SubIOp>>(
        &getContext(), /*isMul*/ false, /*includeFirstMul*/ false,
        getOperation(), &solver);
    (void)walkAndApplyPatterns(getOperation(), std::move(patternsAddModReduce));

    // call CSE here because there may be redundant mod reduce
    // one Value may get mod reduced multiple times in
    // multiple Uses
    //
    // also run annotate-mgmt for lowering
    OpPassManager pipeline("builtin.module");
    pipeline.addPass(createCSEPass());
    pipeline.addPass(mgmt::createAnnotateMgmt());
    (void)runPipeline(pipeline, getOperation());
  }
};

}  // namespace heir
}  // namespace mlir
