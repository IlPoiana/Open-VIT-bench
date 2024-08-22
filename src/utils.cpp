#include "../include/utils.h"

#include "../include/vision_transformer.h"

#include <iostream>
#include <fstream>
#include <assert.h>



void load_cvit(const std::string& path, VisionTransformer& vit) {
    std::ifstream is(path.c_str(), std::ios::binary);
    assert( is.is_open() );

    char intro[4];
    is.read(intro, 4);
    assert( intro[0] == 'C' );
    assert( intro[1] == 'V' );
    assert( intro[2] == 'I' );
    assert( intro[3] == 'T' );

    vit.from_ifstream(is);
    is.close();
}

void store_cvit(const std::string& path, const VisionTransformer& vit) {
    std::ofstream os(path.c_str(), std::ios::binary);
    assert( os.is_open() );

    char intro[4] = {'C', 'V', 'I', 'T'};
    os.write(intro, 4);

    vit.to_ofstream(os);
    os.close();
}

void load_cpic(const std::string& path, PictureBatch& pic) {
    std::ifstream is(path.c_str(), std::ios::binary);
    assert( is.is_open() );

    char intro[4];
    is.read(intro, 4);
    assert( intro[0] == 'C' );
    assert( intro[1] == 'P' );
    assert( intro[2] == 'I' );
    assert( intro[3] == 'C' );

    pic.from_ifstream(is);
    is.close();
}

void store_cpic(const std::string& path, const PictureBatch& pic) {
    std::ofstream os(path.c_str(), std::ios::binary);
    assert( os.is_open() );

    char intro[4] = {'C', 'P', 'I', 'C'};
    os.write(intro, 4);

    pic.to_ofstream(os);
    os.close();
}

void load_cprd(const std::string& path, PredictionBatch& prd) {
    std::ifstream is(path.c_str(), std::ios::binary);
    assert( is.is_open() );

    char intro[4];
    is.read(intro, 4);
    assert( intro[0] == 'C' );
    assert( intro[1] == 'P' );
    assert( intro[2] == 'R' );
    assert( intro[3] == 'D' );

    prd.from_ifstream(is);
    is.close();
}

void store_cprd(const std::string& path, const PredictionBatch& prd) {
    std::ofstream os(path.c_str(), std::ios::binary);
    assert( os.is_open() );

    char intro[4] = {'C', 'P', 'R', 'D'};
    os.write(intro, 4);

    prd.to_ofstream(os);
    os.close();
}
