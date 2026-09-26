// RVK1 dispatch record: output[0..21], inputs[32+24*i], attributes[224..255].
layout(set=0,binding=0,std430) buffer Arena { float data[]; };
layout(set=0,binding=1,std430) readonly buffer Metadata { int records[]; };
layout(push_constant) uniform Push { uint record; } push;
int p(int i) { return records[int(push.record)*256+i]; }
int a(int i) { return p(224+i); }
float af(int i) { return intBitsToFloat(a(i)); }
int t(int slot,int field) { return p(32+24*slot+field); }
int coord(int index,int axis) { return (index/p(12+axis))%p(4+axis); }
int broadcastIndex(int slot,int index) {
    int result=t(slot,0), delta=p(3)-t(slot,1);
    for(int j=0;j<t(slot,1);++j)
        if(t(slot,4+j)>1) result+=coord(index,delta+j)*t(slot,12+j);
    return result;
}
float value(int slot,int index) { return data[broadcastIndex(slot,index)]; }
float invalidValue() { return uintBitsToFloat(0x7fc00000u); }
