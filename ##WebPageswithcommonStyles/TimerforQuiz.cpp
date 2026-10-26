
#include <iostream>
#include <pthread.h>
#include <unistd.h> // for sleep()

// Global variables (standard in C++98 for simple shared state)
int timeLeft = 30;
bool isTimerRunning = false;
pthread_t timerThread;
pthread_mutex_t timerMutex = PTHREAD_MUTEX_INITIALIZER;

// Mock functions
void updateTimerUI() {
    pthread_mutex_lock(&timerMutex);
    std::cout << "Time Left: " << timeLeft << "s" << std::endl;
    pthread_mutex_unlock(&timerMutex);
}

void autoSubmit() {
    std::cout << "Timer expired! Auto-submitting..." << std::endl;
}

// The "Thread Routine" - equivalent to the JS callback
void* timerRoutine(void* arg) {
    while (true) {
        sleep(1); // Sleep for 1 second

        pthread_mutex_lock(&timerMutex);
        if (!isTimerRunning) {
            pthread_mutex_unlock(&timerMutex);
            break; 
        }

        timeLeft--;
        pthread_mutex_unlock(&timerMutex);

        updateTimerUI();

        pthread_mutex_lock(&timerMutex);
        if (timeLeft <= 0) {
            isTimerRunning = false;
            pthread_mutex_unlock(&timerMutex);
            autoSubmit();
            break;
        }
        pthread_mutex_unlock(&timerMutex);
    }
    return NULL;
}

void resetTimer() {
    // 1. Stop existing timer (C++98 manual flag)
    pthread_mutex_lock(&timerMutex);
    isTimerRunning = false;
    pthread_mutex_unlock(&timerMutex);

    // 2. Reset values
    pthread_mutex_lock(&timerMutex);
    timeLeft = 30;
    isTimerRunning = true;
    pthread_mutex_unlock(&timerMutex);

    updateTimerUI();

    // 3. Create the thread (JS: setInterval)
    pthread_create(&timerThread, NULL, timerRoutine, NULL);
}

int main() {
    resetTimer();

    // Keep main thread alive to watch the output
    sleep(35); 
    return 0;
}