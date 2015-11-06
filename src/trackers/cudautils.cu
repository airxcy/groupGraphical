#include <cuda.h>
#include <cublas.h>
#include <cuda_runtime.h>

__device__ void d_HSVtoRGB( unsigned char *r, unsigned char *g, unsigned char *b, float h, float s, float v )
{
    int i;
    float f;
    int p, q, t,vc=v*255;
    vc=vc*(vc>0);
    int indv=vc<255;
    vc = vc*indv+255*(1-indv);
    if( s <= 0.0 ) {
        *r = *g = *b = vc;
        return;
    }
    h /= 60.0;			// sector 0 to 5
    i =  h ;
    f = h - i;			// factorial part of h
    p = v * ( 1.0 - s )*255;
    q = v * ( 1.0 - s * f )*255;
    t = v * ( 1.0 - s * ( 1.0 - f ) )*255;
    p = p*(p>0),q = q*(q>0),t = t*(t>0);
    int indp=p<255,indq=q<255,indt=t<255;
    p = p*indp+255*(1-indp);
    q = q*indq+255*(1-indq);
    t = t*indt+255*(1-indt);
    switch( i ) {
        case 0:
            *r = vc;
            *g = t;
            *b = p;
            break;
        case 1:
            *r = q;
            *g = vc;
            *b = p;
            break;
        case 2:
            *r = p;
            *g = vc;
            *b = t;
            break;
        case 3:
            *r = p;
            *g = q;
            *b = vc;
            break;
        case 4:
            *r = t;
            *g = p;
            *b = vc;
            break;
    case 5:
    default:
            *r = vc;
            *g = p;
            *b = q;
            break;
    }
}
