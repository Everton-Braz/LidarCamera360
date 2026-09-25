#include "RfDetrDecode.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

float sample(const std::vector<float>& plane, int w, int h, float x, float y) {
    x = std::clamp(x, 0.f, float(w - 1)); y = std::clamp(y, 0.f, float(h - 1));
    int x0 = int(x), y0 = int(y), x1 = std::min(x0+1,w-1), y1 = std::min(y0+1,h-1);
    float fx=x-x0, fy=y-y0;
    return (plane[y0*w+x0]*(1-fx)+plane[y0*w+x1]*fx)*(1-fy)
         + (plane[y1*w+x0]*(1-fx)+plane[y1*w+x1]*fx)*fy;
}

int main() {
    try {
        for (int pattern=0; pattern<4; ++pattern) {
            const int mw=7,mh=5,w=173,h=129;
            std::vector<float> plane(mw*mh,-1.f), labels(91,-12.f);
            labels[1]=12.f;
            for(int y=0;y<mh;++y) for(int x=0;x<mw;++x)
                if ((pattern==1 && x<3 && y<3) || pattern==2 || (pattern==3 && (x+y)%3==0)) plane[y*mw+x]=1.f;
            std::vector<nn::TrtTensor> tensors{{"dets",{1,1,4},{.5f,.5f,1.f,1.f}},
                {"labels",{1,1,91},labels},{"masks",{1,1,mh,mw},plane}};
            for(float margin : {0.f,.05f}) {
                std::vector<uint8_t> actual(w*h), original(w*h), expected(w*h);
                for(int y=0;y<h;++y) for(int x=0;x<w;++x)
                    original[y*w+x]=sample(plane,mw,mh,(x+.5f)*mw/w-.5f,(y+.5f)*mh/h-.5f)>0;
                int kernel=std::max(int(margin*.5f*(w+h)),3)|1;
                int radius=margin==0?0:(kernel-1)/2;
                for(int y=0;y<h;++y) for(int x=0;x<w;++x)
                    for(int dy=-radius;dy<=radius;++dy) for(int dx=-radius;dx<=radius;++dx)
                        if(dx*dx+dy*dy<=radius*radius && x+dx>=0 && x+dx<w && y+dy>=0 && y+dy<h)
                            expected[y*w+x] |= original[(y+dy)*w+x+dx];
                rfdetr::decode_people(tensors,w,h,.5f,margin,actual);
                if(actual!=expected) throw std::runtime_error("Cropped interpolation/dilation differs from full-frame reference");
            }
            tensors[1].data[1]=-12.f;
            std::vector<uint8_t> empty(w*h);
            rfdetr::decode_people(tensors,w,h,.5f,0,empty);
            if(std::find(empty.begin(),empty.end(),1)!=empty.end()) throw std::runtime_error("Class threshold ignored");
            tensors[1].data[1]=std::numeric_limits<float>::quiet_NaN();
            bool rejected=false;
            try { rfdetr::decode_people(tensors,w,h,.5f,0,empty); } catch(const std::exception&) { rejected=true; }
            if(!rejected) throw std::runtime_error("Nonfinite output accepted");
        }
        std::cout << "RF-DETR full-frame reference parity: PASS\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
