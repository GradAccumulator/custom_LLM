#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <math.h>
#include <stdint.h>

/*
if seq_len != None:
    stride = int(seq_len*stride_rate)
    plus = []
    for e in encoded:
        if(len(e)>seq_len):
            for i in range(0, len(e)-seq_len+1, stride):
                plus.append(e[i:i+seq_len])
        else:
            plus.append(e)
    encoded = plus
    encoded_x = encoded
    encoded_label = [x[1:] for x in encoded]
12줄
*/

//약 82줄

struct results {
    int16_t* arr_x;
    int16_t* arr_label;
    int act_len;
};

//gcc -shared -o src/c_modules/dlls/seq_windows.dll src/c_modules/seq_windows.c
__declspec(dllexport)
struct results make_seq_windows_with_padding(int16_t** encoded, int max_length, int size, int* lengths, int seq_len, float stride_rate, int pad_id) {
    /*
    입력값은 무조건 int[size][lengths]
    출력값에서 length부분은 seq_len 이하로 바뀜
    따라서 출력은 int[max_length][seq_len]
    */
    //printf("max_length:%d size:%d seq_len:%d stride_rate:%.3f\n", max_length, size, seq_len, stride_rate);
    
    int16_t* out = VirtualAlloc(NULL, max_length*seq_len*sizeof(int16_t), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    int16_t* out_label = VirtualAlloc(NULL, max_length*seq_len*sizeof(int16_t), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if(out==NULL || out_label==NULL) {
        printf("NULL ptr");
        exit(1);
    }
    else {
        memset(out, 0, max_length*seq_len*sizeof(int16_t));
        memset(out_label, 0, max_length*seq_len*sizeof(int16_t));
    }

    int overlap = seq_len * (1 - stride_rate);
    int idx_out_row = 0;
    int idx_out_col = 0;
    int row_cnt = 0;
    int max_row_cnt = max_length;//(int)sqrt(sqrt(max_length)) + 5;
    max_row_cnt /= 10;
    max_row_cnt *= 10;
    struct results res;

    for(int i=0;i<size;i++) {
        int16_t* e = encoded[i];
        row_cnt = 0;

        for(int idx_encoded=0; idx_encoded<lengths[i]; idx_encoded++){
            out[(idx_out_row*seq_len)+(idx_out_col++)] = e[idx_encoded];
            if(idx_out_col==seq_len) {
                //x,label길이 끝남
                out_label[(idx_out_row*seq_len)+(idx_out_col-1)] = pad_id;
                idx_out_col = 0;
                idx_encoded -= overlap-1;
                idx_out_row++;
                if (idx_out_row == max_length) {
                    res.arr_x = out;
                    res.arr_label = out_label;
                    res.act_len = max_length;
                    printf("c function complete\n");
                    return res;
                }
                row_cnt++;
                if (row_cnt>max_row_cnt) {
                    break;
                }
            }
            else if(idx_encoded+1<lengths[i]) {
                //아직 x,label길이 안끝남 + 추가해야할 문장에 단어들이 남음
                out_label[(idx_out_row*seq_len)+(idx_out_col-1)] = e[idx_encoded+1];
            }
            else {
                //아직 x,label길이 안끝남 + 추가해야할 문장에 단어들이 안남음
                // -> x,label에 seq_len 길이 맞춰서 더 추가해야함
                out_label[(idx_out_row*seq_len)+(idx_out_col-1)] = pad_id;
                if (pad_id!=0) {
                    for(int idx=idx_out_col; idx<seq_len; idx++) {
                        out[(idx_out_row*seq_len)+(idx)]       = pad_id;
                        out_label[(idx_out_row*seq_len)+(idx)] = pad_id;
                    }
                }
                idx_out_col = 0;
                idx_out_row++;
                if (idx_out_row == max_length) {
                    res.arr_x = out;
                    res.arr_label = out_label;
                    res.act_len = max_length;
                    printf("c function complete\n");
                    return res;
                }
                row_cnt++;
                if (row_cnt>max_row_cnt) {
                    break;
                }
            }
        }
    }
    res.arr_x = out;
    res.arr_label = out_label;
    res.act_len  = idx_out_row;
    printf("c function complete\n");
    return res;
}


__declspec(dllexport)
void free_results(int16_t* arr, int size) {
    VirtualFree((void*)arr, 0, MEM_RELEASE);
}

/*
이후 tokenizer.py에서
x = [[]]
label = [[]]
res = make_seq_windows_with_padding(~~) 한다음에
for i in range(max_length-res.garbage_len):
    for k in range(seq_len):
        x[i].append(res.arr_x[k])
        label[i].append(res_arr_label[k])
    x.append([])
    label.append([])
free_results(res.arr_x)
free_results(res.arr_label)
*/