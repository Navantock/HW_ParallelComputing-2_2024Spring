#include <iostream>
#include <cstdlib>
#include <fstream>
#include <chrono>
#include <vector>
#include <numeric>


#define BLOCK_ROWS 16
#define BLOCK_COLS 16


__global__ void Init_Labeling(unsigned int* labels, unsigned int* b_labels, const int label_rows, const int label_cols) {
    unsigned int row = (blockIdx.y * BLOCK_ROWS + threadIdx.y) * 2;
    unsigned int col = (blockIdx.x * BLOCK_COLS + threadIdx.x) * 2;
    unsigned int labels_index = row * label_cols + col;

    if (row < label_rows && col < label_cols) {
        labels[labels_index] = labels_index;
        b_labels[labels_index] = labels_index;
    } 
}

__device__ unsigned int Find(unsigned int* labels, unsigned int label) {
    while (labels[label] != label) {
        label = labels[label];
    }
    return label;
}

__device__ void Union(unsigned int* labels, unsigned int a_label_idx, unsigned int b_label_idx) {
    bool end = false;
    do {
        unsigned int a_label = Find(labels, a_label_idx);
        unsigned int b_label = Find(labels, b_label_idx);

        if (a_label < b_label) {
            unsigned int old = atomicMin(&labels[b_label], a_label);
            end = (old == b_label);
            b_label_idx = old;
        } else if (b_label < a_label) {
            unsigned int old = atomicMin(&labels[a_label], b_label);
            end = (old == a_label);
            a_label_idx = old;
        } else {
            end = true;
        }
    } while (!end);
}

__global__ void Merge(unsigned int* img, unsigned int* labels, unsigned int* b_labels,  const int img_rows, const int img_cols) {
    const int label_rows = img_rows;
    const int label_cols = img_cols;
    unsigned int row = (blockIdx.y * BLOCK_ROWS + threadIdx.y) * 2;
    unsigned int col = (blockIdx.x * BLOCK_COLS + threadIdx.x) * 2;
    unsigned int img_index = row * img_cols + col;
    unsigned int labels_index = row * label_cols + col;
    unsigned int b_labels_index = row * label_cols + col;

    if (row < label_rows && col < label_cols) {
        // Foreground pixels
#define CONDITION_B col>0 && row>1 && img[img_index - 2 * img_cols - 1] > 0
#define CONDITION_C row>1 && img[img_index - 2 * img_cols] > 0
#define CONDITION_D col+1<img_cols && row>1 && img[img_index - 2 * img_cols + 1] > 0
#define CONDITION_E col+2<img_cols && row>1 && img[img_index - 2 * img_cols + 2] > 0

#define CONDITION_G col>1 && row>0 && img[img_index - img_cols - 2] > 0
#define CONDITION_H col>0 && row>0 && img[img_index - img_cols - 1] > 0
#define CONDITION_I row>0 && img[img_index - img_cols] > 0
#define CONDITION_J col+1<img_cols && row>0 && img[img_index - img_cols + 1] > 0
#define CONDITION_K col+2<img_cols && row>0 && img[img_index - img_cols + 2] > 0

#define CONDITION_M col>1 && img[img_index - 2] > 0
#define CONDITION_N col>0 && img[img_index - 1] > 0
#define CONDITION_O img[img_index] > 0
#define CONDITION_P col+1<img_cols && img[img_index + 1] > 0

#define CONDITION_R col>0 && row+1<img_rows && img[img_index + img_cols - 1] > 0
#define CONDITION_S row+1<img_rows && img[img_index + img_cols] > 0
#define CONDITION_T col+1<img_cols && row+1<img_rows && img[img_index + img_cols + 1] > 0

// Action 1: No action
#define ACTION_1  
//			// Action 2: New label (the block has foreground pixels and is not connected to anything else)
#define ACTION_2  
            //Action P: Merge with block P
#define ACTION_3 Union(labels, labels_index, labels_index - 2 * img_cols - 2); 
            // Action Q: Merge with block Q
#define ACTION_4 Union(labels, labels_index, labels_index - 2 * img_cols);	
            // Action R: Merge with block R
#define ACTION_5 Union(labels, labels_index, labels_index - 2 * img_cols + 2); 
            // Action S: Merge with block S
#define ACTION_6 Union(labels, labels_index, labels_index - 2);  
            // Action 7: Merge labels of block P and Q
#define ACTION_7 Union(labels, labels_index, labels_index - 2 * img_cols - 2); \
            Union(labels, labels_index, labels_index - 2 * img_cols);			
            //Action 8: Merge labels of block P and R
#define ACTION_8 Union(labels, labels_index, labels_index - 2 * img_cols - 2); \
            Union(labels, labels_index, labels_index - 2 * img_cols + 2);			
            // Action 9 Merge labels of block P and S
#define ACTION_9 Union(labels, labels_index, labels_index - 2 * img_cols - 2); \
            Union(labels, labels_index, labels_index - 2);			
            // Action 10 Merge labels of block Q and R
#define ACTION_10 Union(labels, labels_index, labels_index - 2 * img_cols); \
            Union(labels, labels_index, labels_index - 2 * img_cols + 2);			
            // Action 11: Merge labels of block Q and S
#define ACTION_11 Union(labels, labels_index, labels_index - 2 * img_cols); \
            Union(labels, labels_index, labels_index - 2);			
            // Action 12: Merge labels of block R and S
#define ACTION_12 Union(labels, labels_index, labels_index - 2 * img_cols + 2); \
            Union(labels, labels_index, labels_index - 2);			
            // Action 13: not used
#define ACTION_13 
            // Action 14: Merge labels of block P, Q and S
#define ACTION_14 Union(labels, labels_index, labels_index - 2 * img_cols - 2); \
            Union(labels, labels_index, labels_index - 2 * img_cols); \
            Union(labels, labels_index, labels_index - 2);		
            //Action 15: Merge labels of block P, R and S
#define ACTION_15 Union(labels, labels_index, labels_index - 2 * img_cols - 2); \
            Union(labels, labels_index, labels_index - 2 * img_cols + 2); \
            Union(labels, labels_index, labels_index - 2);			
            //Action 16: labels of block Q, R and S
#define ACTION_16 Union(labels, labels_index, labels_index - 2 * img_cols); \
            Union(labels, labels_index, labels_index - 2 * img_cols + 2); \
            Union(labels, labels_index, labels_index - 2);	

        #include "drag_fg.inc.cuh"

#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5
#undef ACTION_6
#undef ACTION_7
#undef ACTION_8
#undef ACTION_9
#undef ACTION_10
#undef ACTION_11
#undef ACTION_12
#undef ACTION_13
#undef ACTION_14
#undef ACTION_15
#undef ACTION_16


#undef CONDITION_B
#undef CONDITION_C
#undef CONDITION_D
#undef CONDITION_E

#undef CONDITION_G
#undef CONDITION_H
#undef CONDITION_I
#undef CONDITION_J
#undef CONDITION_K

#undef CONDITION_M
#undef CONDITION_N
#undef CONDITION_O
#undef CONDITION_P

#undef CONDITION_R
#undef CONDITION_S
#undef CONDITION_T
    
    // Background pixels
#define CONDITION_B col>0 && row>1 && img[img_index - 2 * img_cols - 1] == 0
#define CONDITION_C row>1 && img[img_index - 2 * img_cols] == 0
#define CONDITION_D col+1<img_cols && row>1 && img[img_index - 2 * img_cols + 1] == 0
#define CONDITION_E col+2<img_cols && row>1 && img[img_index - 2 * img_cols + 2] == 0

#define CONDITION_G col>1 && row>0 && img[img_index - img_cols - 2] == 0
#define CONDITION_H col>0 && row>0 && img[img_index - img_cols - 1] == 0
#define CONDITION_I row>0 && img[img_index - img_cols] == 0
#define CONDITION_J col+1<img_cols && row>0 && img[img_index - img_cols + 1] == 0
#define CONDITION_K col+2<img_cols && row>0 && img[img_index - img_cols + 2] == 0

#define CONDITION_M col>1 && img[img_index - 2] == 0
#define CONDITION_N col>0 && img[img_index - 1] == 0
#define CONDITION_O img[img_index] == 0
#define CONDITION_P col+1<img_cols && img[img_index + 1] == 0

#define CONDITION_R col>0 && row+1<img_rows && img[img_index + img_cols - 1] == 0
#define CONDITION_S row+1<img_rows && img[img_index + img_cols] == 0
#define CONDITION_T col+1<img_cols && row+1<img_rows && img[img_index + img_cols + 1] == 0

// Action 1: No action
#define ACTION_1  
// Action 2: New label (the block has foreground pixels and is not connected to anything else)
#define ACTION_2  
//Action P: Merge with block P
#define ACTION_3 Union(b_labels, b_labels_index, b_labels_index - 2 * label_cols - 2); 
// Action Q: Merge with block Q
#define ACTION_4 Union(b_labels, b_labels_index, b_labels_index - 2 * label_cols);	
// Action R: Merge with block R
#define ACTION_5 Union(b_labels, b_labels_index, b_labels_index - 2 * label_cols + 2); 
// Action S: Merge with block S
#define ACTION_6 Union(b_labels, b_labels_index, b_labels_index - 2);  
// Action 7: Merge labels of block P and Q
#define ACTION_7 Union(b_labels, b_labels_index, b_labels_index - 2 * label_cols - 2); Union(b_labels, b_labels_index, b_labels_index - 2 * label_cols);			
//Action 8: Merge labels of block P and R
#define ACTION_8 Union(b_labels, b_labels_index, b_labels_index - 2 * label_cols - 2); Union(b_labels, b_labels_index, b_labels_index - 2 * label_cols + 2);			
// Action 9 Merge labels of block P and S
#define ACTION_9 Union(b_labels, b_labels_index, b_labels_index - 2 * label_cols - 2); Union(b_labels, b_labels_index, b_labels_index - 2);			
// Action 10 Merge labels of block Q and R
#define ACTION_10 Union(b_labels, b_labels_index, b_labels_index - 2 * label_cols); Union(b_labels, b_labels_index, b_labels_index - 2 * label_cols + 2);			
// Action 11: Merge labels of block Q and S
#define ACTION_11 Union(b_labels, b_labels_index, b_labels_index - 2 * label_cols); Union(b_labels, b_labels_index, b_labels_index - 2);			
// Action 12: Merge labels of block R and S
#define ACTION_12 Union(b_labels, b_labels_index, b_labels_index - 2 * label_cols + 2); Union(b_labels, b_labels_index, b_labels_index - 2);			
// Action 13: not used
#define ACTION_13 
// Action 14: Merge labels of block P, Q and S
#define ACTION_14 Union(b_labels, b_labels_index, b_labels_index - 2 * label_cols - 2); Union(b_labels, b_labels_index, b_labels_index - 2 * label_cols); Union(b_labels, b_labels_index, b_labels_index - 2);		
//Action 15: Merge labels of block P, R and S
#define ACTION_15 Union(b_labels, b_labels_index, b_labels_index - 2 * label_cols - 2); Union(b_labels, b_labels_index, b_labels_index - 2 * label_cols + 2); Union(b_labels, b_labels_index, b_labels_index - 2);			
//Action 16: labels of block Q, R and S
#define ACTION_16 Union(b_labels, b_labels_index, b_labels_index - 2 * label_cols); Union(b_labels, b_labels_index, b_labels_index - 2 * label_cols + 2); Union(b_labels, b_labels_index, b_labels_index - 2);
            
            #include "drag_bg.inc.cuh"

#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5
#undef ACTION_6
#undef ACTION_7
#undef ACTION_8
#undef ACTION_9
#undef ACTION_10
#undef ACTION_11
#undef ACTION_12
#undef ACTION_13
#undef ACTION_14
#undef ACTION_15
#undef ACTION_16


#undef CONDITION_B
#undef CONDITION_C
#undef CONDITION_D
#undef CONDITION_E

#undef CONDITION_G
#undef CONDITION_H
#undef CONDITION_I
#undef CONDITION_J
#undef CONDITION_K

#undef CONDITION_M
#undef CONDITION_N
#undef CONDITION_O
#undef CONDITION_P

#undef CONDITION_R
#undef CONDITION_S
#undef CONDITION_T
    }
}

__global__ void Path_Compression(unsigned int* labels, unsigned int* b_labels, const int img_rows, const int img_cols) {
    const int label_rows = img_rows;
    const int label_cols = img_cols;
    unsigned int row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    unsigned int col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    unsigned int labels_index = row * label_cols + col;
    unsigned int b_labels_index = row * label_cols + col;

    if (row < label_rows && col < label_cols) {
        labels[labels_index] = Find(labels, labels_index);
        b_labels[labels_index] = Find(b_labels, b_labels_index);
    }
}

__global__ void Distribute_Labels(unsigned int* img, unsigned int* labels, unsigned int* b_labels, const int img_rows, const int img_cols) {
    const int label_rows = img_rows;
    const int label_cols = img_cols;
    unsigned row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    unsigned col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    unsigned img_index = row * img_cols + col;
    unsigned labels_index = row * label_cols + col;

    if (row < label_rows && col < label_cols) {
        unsigned int background_label = b_labels[labels_index] + (unsigned int)(label_rows * label_cols);
        unsigned int foreground_label = labels[labels_index];
        labels[labels_index] = img[img_index]==0 ? background_label: foreground_label;

        if (col + 1 < label_cols) {
            labels[labels_index + 1] = img[img_index + 1]==0 ? background_label: foreground_label;
        }
        if (row + 1 < label_rows) {
            labels[labels_index + label_cols] = img[img_index + img_cols]==0 ? background_label: foreground_label;
        }
        if (col + 1 < label_cols && row + 1 < label_rows) {
            labels[labels_index + label_cols + 1] = img[img_index + img_cols + 1]==0 ? background_label: foreground_label;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << "<input_file> <output_file> <test times>\n";
        exit(EXIT_FAILURE);
    }

    const char* input_filename = argv[1];
    const char* output_filename = argv[2];
    int test_times = atoi(argv[3]);

    FILE* input_file = fopen(input_filename, "r");
    if (input_file == NULL) {
        perror("Error opening input file");
        exit(EXIT_FAILURE);
    }

    int rows, cols; 
    if(!fscanf(input_file, "%d %d", &rows, &cols))
    {
        std::cerr << "Error reading file: size\n";
        exit(EXIT_FAILURE);
    }

    unsigned int* img = (unsigned int*)malloc(rows * cols * sizeof(int));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if(!fscanf(input_file, "%d", &img[i * cols + j]))
            {
                std::cerr << "Error reading file: pixels\n";
                exit(EXIT_FAILURE);
            }
        }
    }

    unsigned int* labels = (unsigned int*)malloc(rows * cols * sizeof(unsigned int));
    
    unsigned int* d_img;
    unsigned int* d_labels;
    unsigned int* db_labels;
    cudaMalloc(&d_img, rows * cols * sizeof(int));
    cudaMalloc(&d_labels, rows * cols * sizeof(int));
    cudaMalloc(&db_labels, rows * cols * sizeof(int));
    
    std::vector<double> running_times;
    dim3 grid_size_ = dim3((((cols + 1) / 2) + BLOCK_COLS - 1) / BLOCK_COLS, (((rows + 1) / 2) + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
    dim3 block_size_ = dim3(BLOCK_COLS, BLOCK_ROWS, 1);
    for (int i = 0; i < test_times; i++) {
        cudaDeviceSynchronize();
        auto start_time = std::chrono::high_resolution_clock::now();
        cudaMemcpy(d_img, img, rows * cols * sizeof(unsigned int), cudaMemcpyHostToDevice);
        Init_Labeling<<<grid_size_, block_size_>>>(d_labels, db_labels, rows, cols);
        Merge<<<grid_size_, block_size_>>>(d_img, d_labels, db_labels, rows, cols);
        Path_Compression<<<grid_size_, block_size_>>>(d_labels, db_labels, rows, cols);
        Distribute_Labels<<<grid_size_, block_size_>>>(d_img, d_labels, db_labels, rows, cols);
        cudaMemcpy(labels, d_labels, rows * cols * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> running_time = end_time - start_time;
        running_times.push_back(running_time.count());
    }

    FILE* fp = fopen(output_filename, "w");
    if (fp == NULL) {
        std::cerr << "Error opening file";
        exit(EXIT_FAILURE);
    }

    // First line: print the number of rows and columns
    fprintf(fp, "%d %d\n", rows, cols);

    // Starting from the second line, print the result of the Union Find
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(fp, "%d ", labels[i * cols + j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    fclose(input_file);

    std::cout << "CUDA_CCL Average Time: " << std::accumulate(running_times.begin(), running_times.end(), 0.0) / running_times.size() << "s" << std::endl;

    free(img);
    free(labels);
    cudaFree(d_img);
    cudaFree(d_labels);
    cudaFree(db_labels);

}