#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!("./bindings.rs");
include!("./anec_add.rs");

fn main() {
	unsafe {
		let nn = ane_init_add();
		ane_free(nn);
	}
}
