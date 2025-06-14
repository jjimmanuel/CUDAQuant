#include <curand_kernel.h>
#include <vector>
#include <iostream>

const int threads_per_block = 256;

void checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << file << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(ans) { checkCudaError((ans), __FILE__, __LINE__); }


__global__ void setupCurand(curandState *states, unsigned long seed, int n) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < n)
		curand_init(seed, idx, 0, &states[idx]);
}


__global__ void heston(float *results_v, float *results_s, curandState *states, float s0, float v0, float kappa, 
			float theta, float sigma_v, float rho, float r, float dt, int numSteps, int numPaths) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= numPaths) return;

	curandState localState = states[idx];

	float current_v = v0;
	float current_s = s0;

	results_v[idx * (numSteps + 1) + 0] = current_v;
	results_s[idx * (numSteps + 1) + 0] = current_s;

	for (int i = 0; i < numSteps; ++i) {
		float Z1 = curand_normal(&localState);
		float Z2 = curand_normal(&localState);
		
		float dW_s = sqrt(dt) * Z1;
		float dW_v = sqrt(dt) * (rho * Z1 + sqrt(1.0 - rho *  rho) * Z2);
		
		float next_v = current_v + kappa * (theta - current_v) * dt + sigma_v * sqrt(current_v) * dW_v;
		next_v = fmaxf(0.0f, next_v);
		
		float next_s = current_s * exp((r - 0.5 * current_v) * dt + sqrt(current_v) * dW_s);

		results_v[idx * (numSteps + 1) + i + 1] = next_v;
		results_s[idx * (numSteps + 1) + i + 1] = next_s;

		current_s = next_s;
		current_v = next_v;

	}

	states[idx] = localState;

}


int main() {

	// 1. Simulation parameters
	const int numSteps = 252;
	const int numPaths = 10000;
	const float s0 = 10.0;
	const float v0 = 0.2; 
	const float kappa = 1.0;
	const float theta = 1.0;
	const float sigma_v = 0.15;
	const float rho = 1.0;	
	const float r = 0.03;
	const float dt = 1.0f/numSteps;
	

	// 2. Allocate host and device memory
	float *device_results_v;
	float *device_results_s;
	curandState *device_states;

	std::vector<float> host_results_v(numPaths * (numSteps + 1));
	std::vector<float> host_results_s(numPaths * (numSteps + 1));

	CUDA_CHECK(cudaMalloc(&device_results_v, numPaths * numSteps * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&device_results_s, numPaths * numSteps * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&device_states, numPaths * sizeof(curandState)));

	//3 . Setup curand states
	int blocks = (numPaths + threads_per_block - 1) / threads_per_block;
	setupCurand<<<blocks, threads_per_block>>>(device_states, 0, numPaths);
	CUDA_CHECK(cudaDeviceSynchronize());

	// 4. Launch Heston kernel
	heston<<<blocks, threads_per_block>>>(device_results_v, device_results_s, device_states, 
						s0, v0, kappa, theta, sigma_v, rho, r, dt, numSteps,
						numPaths);
	CUDA_CHECK(cudaDeviceSynchronize());
	
	// 5. Transfer results from Device to Host
	CUDA_CHECK(cudaMemcpy(host_results_v.data(), device_results_v, numPaths * numSteps * sizeof(float), cudaMemcpyDeviceToHost());
	CUDA_CHECK(cudaMemcpy(host_results_s.data(), device_results_s, numPaths * numSteps * sizeof(float), cudaMemcpyDeviceToHost());

	CUDA_CHECK(cudaFree(device_results_v));
	CUDA_CHECK(cudaFree(device_results_s));
	CUDA_CHECK(cudaFree(device_states));

	return 0;

}


	

	

		