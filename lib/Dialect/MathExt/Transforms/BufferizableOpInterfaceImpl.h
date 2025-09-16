#ifndef LIB_DIALECT_MATHEXT_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_
#define LIB_DIALECT_MATHEXT_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_

namespace mlir {

class DialectRegistry;

namespace heir {
namespace math_ext {

void registerBufferizableOpInterfaceExternalModels(DialectRegistry& registry);

}  // namespace math_ext
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_MATHEXT_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H_
