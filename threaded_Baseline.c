//pthreads

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>
#include <pthread.h>

#define NUM_STATES 500
#define NUM_ACTIONS 16
#define ALPHA 0.1
#define GAMMA 0.95
#define NUM_EPISODES 2000
#define BATCH_SIZE 500
#define BATCH_CAPACITY 1000000
#define NUM_STRIDE 4  // Define a stride value
#define EPSILON 0.1 // For epsilon-greedy policy

// Define a macro for the number of threads
#define NUM_THREADS 16

typedef struct {
    int state;
    int action;
    double reward;
    int next_state;
} Experience;

typedef struct {
    Experience* dataset;
    int start_index;
    int end_index;
    double (*q_table)[NUM_ACTIONS];  // Pass Q-table as a pointer to the array
} ThreadData;

typedef enum {
    SEQUENTIAL = 0,
    RANDOM, 
    STRIDE,
} sampling ;

typedef enum  {
    QLEARN = 0,
    SARSA
} algorithm;

algorithm algorithm_type = QLEARN;
sampling sampling_type = SEQUENTIAL;
int num_actions = NUM_ACTIONS;
int num_states = NUM_STATES;

//pthread_mutex_t q_table_mutex = PTHREAD_MUTEX_INITIALIZER;
// Define LCG parameters
 #define LCG_A 1664525
 #define LCG_C 1013904223

 // Custom random number generator
 unsigned int custom_rand(unsigned int *seed) {
     *seed = (*seed * LCG_A + LCG_C);
         return *seed;
	}



void update_q_table(Experience experience, double (*q_table)[NUM_ACTIONS]) {
    int s = experience.state;
    int a = experience.action;
    double r = experience.reward;
    int next_s = experience.next_state;

    // Perform Q-value update
    double max_next_q = 0;
    for (int next_a = 0; next_a < num_actions; next_a++) {
        if (q_table[next_s][next_a] > max_next_q) {
            max_next_q = q_table[next_s][next_a];
        }
    }

    //pthread_mutex_lock(&q_table_mutex);
    q_table[s][a] += ALPHA * (r + GAMMA * max_next_q - q_table[s][a]);
    //pthread_mutex_unlock(&q_table_mutex);
}

// Function to choose the next action based on the epsilon-greedy policy
int sarsa_choose_action(unsigned int *seed, int state, double (*q_table) [num_actions]) {
    double rand_val = (double)custom_rand(seed) / RAND_MAX;
    if (rand_val < EPSILON) {
        // Exploration: choose a random action
        return rand() % num_actions;
    } else {
        // Exploitation: choose the best action based on the Q-table
        int best_action = 0;
        double best_value = q_table[state][0];
        for (int a = 1; a < num_actions; a++) {
            if (q_table[state][a] > best_value) {
                best_value = q_table[state][a];
                best_action = a;
            }
        }
        return best_action;
    }
}

void update_q_table_sarsa(unsigned int *seed, Experience experience, double (*q_table)[num_actions]) {
    int s = experience.state;
    int a = experience.action;
    double r = experience.reward;
    int next_s = experience.next_state;

    // Determine the next action based on the current policy
    int next_a = sarsa_choose_action(seed, next_s, q_table);

    // SARSA Q-value update
    double next_q = q_table[next_s][next_a];
    q_table[s][a] += ALPHA * (r + GAMMA * next_q - q_table[s][a]);
}


void* update_seq_thread(void* thread_data) {
    ThreadData* data = (ThreadData*)thread_data;
    unsigned int seed = 42;

    for (int episode = 0; episode < NUM_EPISODES; episode++) {
        for (int i = data->start_index; i < data->end_index; i++) {
            if (algorithm_type == QLEARN ) 
                update_q_table(data->dataset[i], data->q_table);
            else 
                update_q_table_sarsa(&seed, data->dataset[i], data->q_table);
        }
    }

    pthread_exit(NULL);
}

void* update_rand_thread(void* thread_data) {
    ThreadData* data = (ThreadData*)thread_data;
    unsigned int seed = 42;
    unsigned int rand_seed = 42;

    for (int episode = 0; episode < NUM_EPISODES; episode++) {
        for (int i = data->start_index; i < data->end_index; i++) {
            int random_index = custom_rand(&rand_seed) % (data->end_index - data->start_index) + data->start_index ;
            if (algorithm_type == QLEARN) 
                update_q_table(data->dataset[random_index], data->q_table);
            else 
                update_q_table_sarsa(&seed, data->dataset[random_index], data->q_table);
        }
    }

    pthread_exit(NULL);
}

void* update_stride_thread(void* thread_data) {
    ThreadData* data = (ThreadData*)thread_data;
    unsigned int seed = 42;

    for (int episode = 0; episode < NUM_EPISODES; episode++) {
        for (int stride_idx=0; stride_idx < NUM_STRIDE; stride_idx++) {
            int size = data->end_index - data->start_index;
            for (int i = 0; i < size / NUM_STRIDE; i++) {
                int index = stride_idx + i * NUM_STRIDE;
                if (algorithm_type == QLEARN) 
                    update_q_table(data->dataset[index], data->q_table);
                else 
                    update_q_table_sarsa(&seed, data->dataset[index], data->q_table);
            }
        }
    }

    pthread_exit(NULL);
}



int main(int argc, char *argv[]) {
    // Check if the correct number of arguments is provided
    if (argc != 7) {
        fprintf(stderr, "Usage: %s <filepath> <num_states> <num_actions> <num_samples> <sampling> <algorithm>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Extract command line arguments
    char *filepath = argv[1];
    num_states = atoi(argv[2]);
    num_actions = atoi(argv[3]);
    int num_samples = atoi(argv[4]);
    char *sampling_str = argv[5];
    char *algorithm_str = argv[6];

    // Convert sampling string to enum
    if (strcmp(sampling_str, "SEQUENTIAL") == 0) {
        sampling_type = SEQUENTIAL;
    } else if (strcmp(sampling_str, "RANDOM") == 0) {
        sampling_type = RANDOM;
    } else if (strcmp(sampling_str, "STRIDE") == 0) {
        sampling_type = STRIDE;
    } else {
        fprintf(stderr, "Invalid sampling type: %s\n", sampling_str);
        return EXIT_FAILURE;
    }

    // Convert algorithm string to enum
    if (strcmp(algorithm_str, "QLEARN") == 0) {
        algorithm_type = QLEARN;
    } else if (strcmp(algorithm_str, "SARSA") == 0) {
        algorithm_type = SARSA;
    } else {
        fprintf(stderr, "Invalid algorithm type: %s\n", algorithm_str);
        return EXIT_FAILURE;
    }

    // Your program logic goes here, using the extracted variables

    // Print the extracted values for demonstration purposes
    printf("Filepath: %s\n", filepath);
    printf("Num States: %d\n", num_states);
    printf("Num Actions: %d\n", num_actions);
    printf("Num Samples: %d\n", num_samples);
    printf("Sampling Type: %d\n", sampling_type);
    printf("Algorithm Type: %d\n", algorithm_type);

    // clock_t start_time = clock();

    FILE *file = fopen(filepath, "r");
    if (file == NULL) {
        perror("Error opening the data file");
        return 1;
    }

   // srand(time(NULL)); // Seed the random number generator
    // Dynamic memory allocation for the dataset
    Experience* dataset = (Experience*)malloc(BATCH_CAPACITY * sizeof(Experience));
    if (dataset == NULL) {
        perror("Error allocating memory for dataset");
        fclose(file);
        return 1;
    }

    Experience experience;
    
    int num_s = 0;
    while (num_s < num_samples && fscanf(file, "%d %d %lf %d", &experience.state, &experience.action, &experience.reward, &experience.next_state) != EOF) {
        dataset[num_s] = experience;
        num_s++;
    }

    fclose(file);

    // Divide the dataset into chunks
    int chunk_size = num_samples / NUM_THREADS;
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];

    // Initialize Q-tables for each thread
    double q_tables[NUM_THREADS][NUM_STATES][NUM_ACTIONS];

    for (int i = 0; i < NUM_THREADS; i++) {
        for (int state = 0; state < num_states; state++) {
            for (int action = 0; action < num_actions; action++) {
                q_tables[i][state][action] = 0.0;
            }
        }
    }

    clock_t start_time = clock();

    void* (*update_q_table_thread_func)(void*);

    // Assign the appropriate function based on the sampling type
    switch (sampling_type) {
        case SEQUENTIAL:
            update_q_table_thread_func = update_seq_thread;
            break;
        case RANDOM:
            update_q_table_thread_func = update_rand_thread;
            break;
        case STRIDE:
            update_q_table_thread_func = update_stride_thread;
            break;
        default:
            fprintf(stderr, "Invalid sampling type\n");
            return -1;
    }


    // Create threads and perform Q-learning updates in parallel - sequential
    
    for (int batch_window = 0; batch_window < NUM_THREADS; batch_window++) {
        
                thread_data[batch_window].dataset = dataset;
                thread_data[batch_window].start_index =  batch_window * chunk_size;
                thread_data[batch_window].end_index = (batch_window + 1) * chunk_size;
                thread_data[batch_window].q_table = q_tables[batch_window];
                
                pthread_create (&threads[batch_window], NULL, update_q_table_thread_func, (void*)&thread_data[batch_window]);
                //update_q_table(dataset[i], q_tables[batch_window]);    
    }


    // Join threads to wait for their completion
    for (int batch_window = 0; batch_window < NUM_THREADS; batch_window++) {
        pthread_join(threads[batch_window], NULL);
    }

    clock_t end_time = clock();
    double total_time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    // Print Q-tables for each thread
    for (int i = 0; i < NUM_THREADS; i++) {
        printf("Q-table for Thread %d:\n", i);
        for (int state = 0; state < num_states; state++) {
            for (int action = 0; action < num_actions; action++) {

                printf("Q(%d, %d) = %f\n", state, action, q_tables[i][state][action]);
            }
        }
        printf("\n");
    }

    // Free allocated memory for the dataset
    free(dataset);

    // clock_t end_time = clock();
    // double total_time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Total time taken for Q-learning updates: %f seconds\n", total_time_taken);

    return 0;
}




