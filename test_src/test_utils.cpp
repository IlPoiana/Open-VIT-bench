#include "../include/utils.h"

#include <iostream>

using namespace std;

#define bool_plot(x) (x==true ? "True" : "False")

int main() {
    cout << "Test Utils" << endl;

    VisionTransformer vit;
    PictureBatch pic;
    PredictionBatch prd;

    load_cvit("../test_files/test_utils.cvit", vit);
    load_cpic("../test_files/test_utils.cpic", pic);
    load_cprd("../test_files/test_utils.cprd", prd);



    cout << "### vit" << endl;
    cout << "   num_classes: " << vit.get_num_classes() << endl;
    switch ( vit.get_global_pool() ) {
        case pool_token: cout << "   global_pool: token" << endl;
        break;
        case pool_avg: cout << "   global_pool: avg" << endl;
        break;
        case pool_avgmax: cout << "   global_pool: avgmax" << endl;
        break;
        case pool_max: cout << "   global_pool: max" << endl;
        break;
        default: cout << "   global_pool:" << endl;
        break;
    }
    cout << "   embed_dim: " << vit.get_embed_dim() << endl;
    cout << "   depth: " << vit.get_depth() << endl;

    cout << "   has_class_token: " << bool_plot(vit.get_has_class_token()) << endl;
    cout << "   num_reg_tokens: " << vit.get_num_reg_tokens() << endl;
    cout << "   num_prefix_tokens: " << vit.get_num_prefix_tokens() << endl;
    cout << "   no_embed_class: " << bool_plot(vit.get_no_embed_class()) << endl;

    cout << "   use_pos_embed: " << bool_plot(vit.get_use_pos_embed()) << endl;
    cout << "   use_pre_norm: " << bool_plot(vit.get_use_pre_norm()) << endl;
    cout << "   use_fc_norm: " << bool_plot(vit.get_use_fc_norm()) << endl;
    cout << "   dynamic_img_size: " << bool_plot(vit.get_dynamic_img_size()) << endl;



    cout << "### pic" << endl;
    pic.print();
    cout << "### prd" << endl;
    prd.print();



    store_cvit("../test_files/test_utils.cvit", vit);
    store_cpic("../test_files/test_utils.cpic", pic);
    store_cprd("../test_files/test_utils.cprd", prd);

    return 0;
}
