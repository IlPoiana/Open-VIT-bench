#ifndef __UTILS_H__
#define __UTILS_H__

#include <iostream>

#include "vision_transformer.h"



void load_cvit(const std::string& path, VisionTransformer& vit);

void store_cvit(const std::string& path, const VisionTransformer& vit);

void load_cpic(const std::string& path, PictureBatch& pic);

void store_cpic(const std::string& path, const PictureBatch& pic);

void load_cprd(const std::string& path, PredictionBatch& prd);

void store_cprd(const std::string& path, const PredictionBatch& prd);



#endif // __UTILS_H__
