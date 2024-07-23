#ifndef LIB_DIALECT_CGGI_IR_CGGIATTRIBUTES_TD_
#define LIB_DIALECT_CGGI_IR_CGGIATTRIBUTES_TD_

include "lib/Dialect/CGGI/IR/CGGIDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/CommonAttrConstraints.td"

class CGGI_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<CGGI_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";
}

def CGGI_CGGIParams : AttrDef<CGGI_Dialect, "CGGIParams"> {
  let mnemonic = "cggi_params";

  // TODO(#276): migrate the gadget params
  // to lwe dialect?
  let parameters = (ins
    "::mlir::heir::lwe::RLWEParamsAttr": $rlweParams,
    "unsigned": $bsk_noise_variance,
    "unsigned": $bsk_gadget_base_log,
    "unsigned": $bsk_gadget_num_levels,
    "unsigned": $ksk_noise_variance,
    "unsigned": $ksk_gadget_base_log,
    "unsigned": $ksk_gadget_num_levels
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

def CGGIGate_Attr : CGGI_Attr<"CGGIGate", "cggi_gate"> {
  let summary = "An Attribute containing an array of strings to store bool gates";

  let description = [{
    Attribute to store what kind of Boolean gate we are working on
  }];

  let parameters = (ins
      ArrayRefParameter<"mlir::StringAttr">: $gate);

  let assemblyFormat = "`<` $gate `>`";
}




#endif  // LIB_DIALECT_CGGI_IR_CGGIATTRIBUTES_TD_
