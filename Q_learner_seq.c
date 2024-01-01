// #include <stdio.h>
// #include <stdlib.h>
// #include <stdint.h>
// #include <inttypes.h>
// #include <time.h>
// #include <pthread.h>

// #define NUM_STATES 16
// #define NUM_ACTIONS 4
// #define ALPHA 0.1
// #define GAMMA 0.95
// #define NUM_EPISODES 1000
// #define BATCH_SIZE 500
// #define BATCH_CAPACITY 1000000
// #define NUM_THREADS 16  // Number of threads


// double q_table[NUM_STATES][NUM_ACTIONS] = {{0.0}};
// //pthread_mutex_t mutex;

// typedef struct {
//     int state;
//     int action;
//     double reward;
//     int next_state;
// } Experience;

// typedef struct {
//     Experience *dataset;
//     int start_index;
//     int end_index;
// } ThreadArg;

// void update_q_table(Experience experience) {
//     int s = experience.state;
//     int a = experience.action;
//     double r = experience.reward;
//     int next_s = experience.next_state;

//     double max_next_q = 0;
//     for (int next_a = 0; next_a < NUM_ACTIONS; next_a++) {
//         if (q_table[next_s][next_a] > max_next_q) {
//             max_next_q = q_table[next_s][next_a];
//         }
//     }
//     q_table[s][a] += ALPHA * (r + GAMMA * max_next_q - q_table[s][a]);
// }

// void* thread_function(void *arg) {
//     ThreadArg *thread_arg = (ThreadArg *)arg;
//     Experience *dataset = thread_arg->dataset;

//     for (int i = thread_arg->start_index; i < thread_arg->end_index; i++) {
//         //pthread_mutex_lock(&mutex);
//         update_q_table(dataset[i]);
//         //pthread_mutex_unlock(&mutex);
//     }

//     return NULL;
// }

// int main() {
//     FILE *file = fopen("/home/kailash/Downloads/Datasets/FrozenLake_trajectories_1m.txt", "r");
//     if (file == NULL) {
//         perror("Error opening the data file");
//         return 1;
//     }

//     Experience* dataset = (Experience*)malloc(BATCH_CAPACITY * sizeof(Experience));
//     if (dataset == NULL) {
//         perror("Error allocating memory for dataset");
//         fclose(file);
//         return 1;
//     }

//     int num_samples = 0;
//     Experience experience;

//     while (fscanf(file, "%d %d %lf %d", &experience.state, &experience.action, &experience.reward, &experience.next_state) != EOF) {
//         dataset[num_samples] = experience;
//         num_samples++;
//     }

//     fclose(file);

//     if (num_samples < BATCH_SIZE * NUM_THREADS) {
//         fprintf(stderr, "Insufficient data for the number of threads\n");
//         free(dataset);
//         return 1;
//     }

//     pthread_t threads[NUM_THREADS];
//     ThreadArg thread_args[NUM_THREADS];
//     //pthread_mutex_init(&mutex, NULL);

//     clock_t start_time = clock();

//     for (int episode = 0; episode < NUM_EPISODES; episode++) {
//         int mini_batch = num_samples / (BATCH_SIZE * NUM_THREADS);

//         for (int batch_window = 0; batch_window < mini_batch; batch_window++) {
//             for (int t = 0; t < NUM_THREADS; t++) {
//                 int start_index = batch_window * BATCH_SIZE * NUM_THREADS + t * BATCH_SIZE;
//                 int end_index = start_index + BATCH_SIZE;
//                 thread_args[t] = (ThreadArg){.dataset = dataset, .start_index = start_index, .end_index = end_index};
//                 pthread_create(&threads[t], NULL, thread_function, (void *)&thread_args[t]);
//             }

//             for (int t = 0; t < NUM_THREADS; t++) {
//                 pthread_join(threads[t], NULL);
//             }
//         }
//     }

//     clock_t end_time = clock();
//     double total_time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

//     printf("Learned Q-table:\n");
//     for (int state = 0; state < NUM_STATES; state++) {
//         for (int action = 0; action < NUM_ACTIONS; action++) {
//             printf("Q(%d, %d) = %f\n", state, action, q_table[state][action]);
//         }
//     }

//     printf("Total time taken for Q-learning updates: %f seconds\n", total_time_taken);

//     free(dataset);
//     //pthread_mutex_destroy(&mutex);

//     return 0;
// }




#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <pthread.h>

#define NUM_STATES 16
#define NUM_ACTIONS 4
#define ALPHA 0.1
#define GAMMA 0.95
#define NUM_EPISODES 2000
#define BATCH_SIZE 500
#define BATCH_CAPACITY 20000000
#define NUM_THREADS 16  // Number of threads

double q_table[NUM_STATES][NUM_ACTIONS] = {{0.0}};
// pthread_mutex_t mutex;

typedef struct {
    int state;
    int action;
    double reward;
    int next_state;
} Experience;

typedef struct {
    Experience *dataset;
    int start_index;
    int end_index;
} ThreadArg;

void update_q_table(Experience experience) {
    int s = experience.state;
    int a = experience.action;
    double r = experience.reward;
    int next_s = experience.next_state;

    double max_next_q = 0;
    for (int next_a = 0; next_a < NUM_ACTIONS; next_a++) {
        if (q_table[next_s][next_a] > max_next_q) {
            max_next_q = q_table[next_s][next_a];
        }
    }
    q_table[s][a] += ALPHA * (r + GAMMA * max_next_q - q_table[s][a]);
}

void* thread_function(void *arg) {
    ThreadArg *thread_arg = (ThreadArg *)arg;
    Experience *dataset = thread_arg->dataset;

    for (int episode = 0; episode < NUM_EPISODES; episode++) {
        int start_index = thread_arg->start_index;
        int end_index = thread_arg->end_index;  // Corrected line

        for (int i = start_index; i < end_index; i++) {
            update_q_table(dataset[i]);
        }
    }

    return NULL;
}



int main() {
    FILE *file = fopen("/home/upmem0013/kailashg26/Kailash_ISPASS2024/Datasets/Taxi_v3_experiences_5m.txt", "r");
    if (file == NULL) {
        perror("Error opening the data file");
        return 1;
    }

    Experience* dataset = (Experience*)malloc(BATCH_CAPACITY * sizeof(Experience));
    if (dataset == NULL) {
        perror("Error allocating memory for dataset");
        fclose(file);
        return 1;
    }

    int num_samples = 0;
    Experience experience;

    while (fscanf(file, "%d %d %lf %d", &experience.state, &experience.action, &experience.reward, &experience.next_state) != EOF) {
        dataset[num_samples] = experience;
        num_samples++;
    }

    fclose(file);

    if (num_samples < BATCH_SIZE * NUM_THREADS) {
        fprintf(stderr, "Insufficient data for the number of threads\n");
        free(dataset);
        return 1;
    }

    pthread_t threads[NUM_THREADS];
    ThreadArg thread_args[NUM_THREADS];
    // pthread_mutex_init(&mutex, NULL);

    clock_t start_time = clock();

    int total_batch_size = num_samples / NUM_THREADS;
    for (int t = 0; t < NUM_THREADS; t++) {
        int start_index = t * total_batch_size;
        int end_index = start_index + total_batch_size;
        thread_args[t] = (ThreadArg){.dataset = dataset, .start_index = start_index, .end_index = end_index};
        pthread_create(&threads[t], NULL, thread_function, (void *)&thread_args[t]);
    }

    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }

    clock_t end_time = clock();
    double total_time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Learned Q-table:\n");
    for (int state = 0; state < NUM_STATES; state++) {
        for (int action = 0; action < NUM_ACTIONS; action++) {
            printf("Q(%d, %d) = %f\n", state, action, q_table[state][action]);
        }
    }

    printf("Total time taken for Q-learning updates: %f seconds\n", total_time_taken);

    free(dataset);
    // pthread_mutex_destroy(&mutex);

    return 0;
}
