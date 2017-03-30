#include <immintrin.h>
#include <math.h> // you will probably not ned this header

__m256d ONES;

void pow_avx_init () {
    // perform initialization here
    ONES=_mm256_set1_pd(1.0);
}

void print_vector(__m256d vec){
    double buf[4];
    _mm256_storeu_pd(buf,vec);
    printf("%1.0f %1.0f %1.0f %1.0f\n",buf[0], buf[1], buf[2], buf[3]);
}

void print_vector_log(__m256d vec){
    double buf[4];
    _mm256_storeu_pd(buf,vec);
    for(int i=0; i<4; ++i){
        buf[i]=log(buf[i])/log(1.0000001);
    }
    printf("%1.0f %1.0f %1.0f %1.0f\n", buf[0], buf[1], buf[2], buf[3]);
}

#define INIT_VI_MASKS(_vi_name, _u_name)\
    __m256i _vi_name=_mm256_set_epi64x (_u_name<<60, _u_name<<61, _u_name<<62, _u_name<<63);

#define CREATE_MASK(_mask_name, _vi_name)\
    __m256d _mask_name=_mm256_castsi256_pd(_vi_name);

#define DO_THE_THING(_base_name, _result_name, _mask_name){\
    _base_name=_mm256_blendv_pd(ONES, _base_name, _mask_name);\
    _result_name=_base_name;\
    __m256d tmp=_base_name;\
    tmp=_mm256_permute_pd(_base_name, 0x5);\
    _result_name=_mm256_mul_pd(tmp, _base_name);\
    tmp=_mm256_permute2f128_pd(_result_name, _result_name, 0x1);\
    _result_name=_mm256_mul_pd(tmp, _result_name);\
}

#define INIT_BASE(_base_name)\
    _base_name=_mm256_mul_pd(_base_name, _base_name);\
    _base_name=_mm256_mul_pd(_base_name, _base_name);\
    _base_name=_mm256_mul_pd(_base_name, _base_name);\

#define FINISH_BASE(_base_name)\
    var_x_tmp=_mm256_blend_pd(_base_name, ONES, 0x1);\
    _base_name=_mm256_mul_pd(_base_name, var_x_tmp);\
    var_x_tmp=_mm256_blend_pd(_base_name, ONES, 0x3);\
    _base_name=_mm256_mul_pd(_base_name, var_x_tmp);\
    var_x_tmp=_mm256_blend_pd(_base_name, ONES, 0x7);\
    _base_name=_mm256_mul_pd(_base_name, var_x_tmp);\

double pow_avx (double x, uint32_t n_int) {
    u_int64_t u_mask1=0xF; u_mask1&=n_int; n_int=n_int>>4;
    u_int64_t u_mask2=0xF; u_mask2&=n_int; n_int=n_int>>4;
    u_int64_t u_mask3=0xF; u_mask3&=n_int; n_int=n_int>>4;
    u_int64_t u_mask4=0xF; u_mask4&=n_int; n_int=n_int>>4;
    u_int64_t u_mask5=0xF; u_mask5&=n_int; n_int=n_int>>4;
    u_int64_t u_mask6=0xF; u_mask6&=n_int; n_int=n_int>>4;
    u_int64_t u_mask7=0xF; u_mask7&=n_int; n_int=n_int>>4;
    u_int64_t u_mask8=0xF; u_mask8&=n_int; n_int=n_int>>4;

    INIT_VI_MASKS(vi_mask1, u_mask1);
    INIT_VI_MASKS(vi_mask2, u_mask2);
    INIT_VI_MASKS(vi_mask3, u_mask3);
    INIT_VI_MASKS(vi_mask4, u_mask4);
    INIT_VI_MASKS(vi_mask5, u_mask5);
    INIT_VI_MASKS(vi_mask6, u_mask6);
    INIT_VI_MASKS(vi_mask7, u_mask7);
    INIT_VI_MASKS(vi_mask8, u_mask8);

    CREATE_MASK(mask1, vi_mask1);
    CREATE_MASK(mask2, vi_mask2);
    CREATE_MASK(mask3, vi_mask3);
    CREATE_MASK(mask4, vi_mask4);
    CREATE_MASK(mask5, vi_mask5);
    CREATE_MASK(mask6, vi_mask6);
    CREATE_MASK(mask7, vi_mask7);
    CREATE_MASK(mask8, vi_mask8);

    __m256d var_x_tmp;

    // Define BASE_1
    __m256d base_1=_mm256_set1_pd(x);

    // Define BASE_2
    __m256d base_2=_mm256_mul_pd(base_1, base_1);
    INIT_BASE(base_2);

    // Define BASE_3
    __m256d base_3=_mm256_mul_pd(base_2, base_2);
    INIT_BASE(base_3);

    // Define BASE_4
    __m256d base_4=_mm256_mul_pd(base_3, base_3);
    INIT_BASE(base_4);

    // Define BASE_5
    __m256d base_5=_mm256_mul_pd(base_4, base_4);
    INIT_BASE(base_5);

    // Define BASE_6
    __m256d base_6=_mm256_mul_pd(base_5, base_5);
    INIT_BASE(base_6);

    // Define BASE_7
    __m256d base_7=_mm256_mul_pd(base_6, base_6);
    INIT_BASE(base_7);

    // Define BASE_8
    __m256d base_8=_mm256_mul_pd(base_7, base_7);
    INIT_BASE(base_8);

    FINISH_BASE(base_1);
    FINISH_BASE(base_2);
    FINISH_BASE(base_3);
    FINISH_BASE(base_4);
    FINISH_BASE(base_5);
    FINISH_BASE(base_6);
    FINISH_BASE(base_7);
    FINISH_BASE(base_8);

    __m256d result1;
    __m256d result2;
    __m256d result3;
    __m256d result4;
    __m256d result5;
    __m256d result6;
    __m256d result7;
    __m256d result8;

    DO_THE_THING(base_1, result1, mask1);
    DO_THE_THING(base_2, result2, mask2);
    DO_THE_THING(base_3, result3, mask3);
    DO_THE_THING(base_4, result4, mask4);
    DO_THE_THING(base_5, result5, mask5);
    DO_THE_THING(base_6, result6, mask6);
    DO_THE_THING(base_7, result7, mask7);
    DO_THE_THING(base_8, result8, mask8);

    __m256d final_result=ONES;
    final_result=_mm256_mul_pd(final_result, result1);
    final_result=_mm256_mul_pd(final_result, result2);
    final_result=_mm256_mul_pd(final_result, result3);
    final_result=_mm256_mul_pd(final_result, result4);
    final_result=_mm256_mul_pd(final_result, result5);
    final_result=_mm256_mul_pd(final_result, result6);
    final_result=_mm256_mul_pd(final_result, result7);
    final_result=_mm256_mul_pd(final_result, result8);

    double d_result=0;
    __m128d sse_result=_mm256_extractf128_pd(final_result, 0);
    _mm_store_sd(&d_result, sse_result);
    return d_result;
}
