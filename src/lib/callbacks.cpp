#include <iostream>
#include "callbacks.h"
using namespace cv;


void bilateral_threshold(int event, void* userdata)
{
    Data* d = (Data*) userdata;
    Mat* dst = d->mats[1];

    int* dim = (d->ints[0]);
    int* sigma_space = d->ints[1];
    int* sigma_color = (d->ints[2]);

    bilateralFilter(*(d->mats[0]), *dst, *dim, static_cast<double>(*sigma_space)/10, static_cast<double>(*sigma_color)/10);
    imshow( d->winname, *dst );
}

void gaussian_threshold(int event, void* userdata) {
    Data* d = (Data*) userdata;
    Mat dst = (d->mats[0])->clone();

    int* dim = (d->ints[0]);

    GaussianBlur(*(d->mats[0]), dst, Size{*dim * 2 + 1, *dim * 2 + 1}, 0); //disgusting hack cit. Bjarne

    imshow( d->winname, *(d->mats[1]) );
}

void meanshift_trackbar(int event, void* userdata) {
    Data* d = (Data*) userdata;
    Mat* src = d->mats[0];
    Mat* dst = d->mats[1];

    int* sp = d->ints[0];
    int* sr = d->ints[1];
    int* maxlevel = d->ints[2];

    pyrMeanShiftFiltering(*src, *dst, static_cast<double>(*sp)/10, static_cast<double>(*sr)/10, *maxlevel);
    imshow( d->winname, *dst );
}