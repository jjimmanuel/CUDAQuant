# include <curand_kernel.h>
# include <vector>
# include <iostream>

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

__global__ void vasicek(float *results, curandState *states, int numSteps, float dt,
			float a, float b, float sigma, float r0) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= numPaths) return;

	curandState localState = states[idx]; 
	
	float r = r0;
	results[idx * numSteps] = r;

	for (int i = 1; i < numSteps; i++) {
		float Z = curand_normal(&localState);
		r = b + (r - b) * expf(-a * dt) + sigma * sqrtf((1.0f - expf(-2.0f * a * dt)) / (2.0f * a)) * Z;
		results[idx * numSteps + i] = r;
	}

	states[idx] = localState;
}

int main() {

	// 1. Simulation parameters
	const int numPaths = 10000;
	const int numSteps = 252;
	const float T = 1.0;
	const float dt = T/numSteps;
	const float a = 0.1;
	const float b = 0.05;
	const float sigma = 0.15;
	const float r0 = 0.03;

	// 2. Allocate device and host memory	
	float *device_results;
	curandState *device_states;

	std::vector<float> host_results(numPaths * numSteps);

	CUDA_CHECK(cudaMalloc(&device_results, numPaths * numSteps * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&device_states, numPaths * sizeof(curandState)));

	// 3. Setup curand states
	int blocks = (numPaths + threads_per_block - 1) / threads_per_block;
	setupCurand<<<blocks, threads_per_block>>>(device_states, 0, numPaths);
	CUDA_CHECK(cudaDeviceSynchronize());

	// 4. Lauch vasicek kernel
	vasicek<<<blocks, threads_per_block>>>(device_results, device_states, numSteps, dt,
						a, b, sigma, r0);

	CUDA_CHECK(cudaDeviceSynchronize());

	// 5. Transfer results from Device to Host
	CUDA_CHECK(cudaMemcpy(host_results.data(), device_results, numPaths * numSteps * sizeof(float), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(device_results));
	CUDA_CHECK(cudaFree(device_states));

	return 0;

}
	
		

