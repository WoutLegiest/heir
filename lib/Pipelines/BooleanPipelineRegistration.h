#ifndef LIB_PIPELINES_BOOLEANPIPELINEREGISTRATION_H_
#define LIB_PIPELINES_BOOLEANPIPELINEREGISTRATION_H_

#include <string>

#include "llvm/include/llvm/Support/CommandLine.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"     // from @llvm-project
#include "mlir/include/mlir/Pass/PassOptions.h"     // from @llvm-project
#include "mlir/include/mlir/Pass/PassRegistry.h"    // from @llvm-project

namespace mlir::heir {

struct TosaToBooleanTfheOptions
    : public PassPipelineOptions<TosaToBooleanTfheOptions> {
  PassOptions::Option<bool> abcFast{*this, "abc-fast",
                                    llvm::cl::desc("Run abc in fast mode."),
                                    llvm::cl::init(false)};

  PassOptions::Option<int> unrollFactor{
      *this, "unroll-factor",
      llvm::cl::desc("Unroll loops by a given factor before optimizing. A "
                     "value of zero (default) prevents unrolling."),
      llvm::cl::init(0)};

  PassOptions::Option<std::string> entryFunction{
      *this, "entry-function", llvm::cl::desc("Entry function to secretize"),
      llvm::cl::init("main")};
};

void tosaToCGGIPipelineBuilder(OpPassManager &pm,
                               const TosaToBooleanTfheOptions &options,
                               const std::string &yosysFilesPath,
                               const std::string &abcPath,
                               bool abcBooleanGates);

void tosaToArithPipelineBuilder(OpPassManager &pm);

void registerTosaToBooleanTfhePipeline(const std::string &yosysFilesPath,
                                       const std::string &abcPath);

void registerTosaToBooleanFpgaTfhePipeline(const std::string &yosysFilesPath,
                                           const std::string &abcPath);

void registerTosaToArithPipeline();

void registerTosaToJaxitePipeline(const std::string &yosysFilesPath,
                                  const std::string &abcPath);

}  // namespace mlir::heir

#endif  // LIB_PIPELINES_BOOLEANPIPELINEREGISTRATION_H_
