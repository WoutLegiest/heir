#ifndef LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLATTRIBUTES_TD_
#define LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLATTRIBUTES_TD_

include "lib/Dialect/TfheRustBool/IR/TfheRustBoolDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/CommonAttrConstraints.td"

class TfheRustBool_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<TfheRustBool_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";
}

def TfheRustBoolGate_Attr : TfheRustBool_Attr<"TfheRustBoolGate", "tfhe_rust_bool_gate"> {
  let summary = "An Attribute containing an array of strings to store bool gates";

  let description = [{
    Attribute to store what kind of Boolean gate we are working on
  }];

  let parameters = (ins
      ArrayRefParameter<"mlir::StringAttr">: $gate);

  let assemblyFormat = "`<` $gate `>`";
}




#endif  // LIB_DIALECT_TFHERUSTBOOL_IR_TFHERUSTBOOLATTRIBUTES_TD_
