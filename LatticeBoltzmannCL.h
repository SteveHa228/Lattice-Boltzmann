#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <omp.h>

#include "device_info.hpp"

#ifdef __APPLE__
    #include <OpenCL/opencl.h>
    #include <unistd.h>
#else 
    #include "CL/cl.h"
#endif

extern int output_device_info(cl_device_id );

//variable 
//static cl_int err;
static cl_command_queue queue;
static cl_context context;
static cl_device_id device;
static cl_program program;

static inline int divup(int a, int b)
{
    return (a + b - 1) / b;
}

static inline int roundup(int a, int b)
{
    return divup(a, b) * b;
}

static inline void check(cl_int err, const char* context)
{
    if(err != CL_SUCCESS){
        std::cerr << "OpenCL error: " << context << ": " << err << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static inline void check_build(cl_program program, cl_device_id device, cl_int err)
{
    if(err == CL_BUILD_PROGRAM_FAILURE)
    {
        std::cerr << "OpenCL build failed: " << std::endl;
        size_t len;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        std::cout << "len : " << len << std::endl;
        char log[len];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, &log, NULL);
        //std::cout << log << std::endl;
        printf("%s\n", log);
        //delete[]  log; 
        std::exit(EXIT_FAILURE);
    }else if(err != CL_SUCCESS)
    {
        std::cerr << "OpenCL build failed: " << err << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

void initKernel(cl_command_queue& queue, cl_context& context, cl_device_id& device)
{
    cl_int err;
    cl_uint numPlatforms;
    CHECK(clGetPlatformIDs(0, NULL, &numPlatforms));
    cl_platform_id platform[numPlatforms];
    CHECK(clGetPlatformIDs(numPlatforms, platform, NULL));
    //cl_device_id device;
    // Secure a GPU
    for(int i = 0; i < numPlatforms; ++i)
    {
        // Count the number of devices in the platform
        cl_uint num_devices;
        err = clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        //checkError(err, "Finding devices");

        // Get the device IDs
        cl_device_id devices[num_devices];
        err = clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
        //checkError(err, "Getting devices");
        printf("Number of devices: %d\n", num_devices);

        char string[1240] = {0};

        // Investigate each device
        for (int j = 0; j < num_devices; j++)
        {
            printf("\t-------------------------\n");

            // Get device name
            err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(string), &string, NULL);
            //checkError(err, "Getting device name");
            printf("\t\tName: %s\n", string);
            if(strstr(string, "AMD")!=NULL){
                //err = clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);
                device = devices[j];
                break;
            }
        }
    }
    output_device_info(device);
    //CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check(err, "clCreateContext");
#ifdef CL_VERSION_2_0
    queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
#else
    queue = clCreateCommandQueue(context, device, 0, &err);
#endif
    check(err, "clCreateCommandQueue");
}

cl_program LoadKernelFromFile(cl_context& context, cl_int& err)
{
 /**** Load kernel from file ***/
    FILE* programHandle;
    size_t programSize, kernelSourceSize;
    char *programBuffer, *kernelSource;

    // get size of kernel source
    programHandle = fopen("kernel.cl", "r");
    fseek(programHandle, 0, SEEK_END);
    programSize = ftell(programHandle);
    rewind(programHandle);

    // read kernel source into buffer
    programBuffer = (char*) malloc(programSize + 1);
    programBuffer[programSize] = '\0';
    fread(programBuffer, sizeof(char), programSize, programHandle);
    fclose(programHandle);

    //printf("programBuffer = %s\n", programBuffer);

    // create program from buffer
    cl_program program = clCreateProgramWithSource(context, 1,
            (const char**) &programBuffer, NULL, &err);
    free(programBuffer);

    /**** End  ****/
    return program;
}

void initializeCL(const int N, const int Q, const double DENSITY, const double LID_VELOCITY, 
                double *ex, double *ey, int *oppos, double *wt,
                double *rho, double *ux, double *uy, double* sigma, 
                double *f, double *feq, double *f_new)
{
    cl_int err;
    initKernel(queue, context, device);

    //Compile kernel
    program = LoadKernelFromFile(context, err);
    check(err, "clCreateProgramWithSource");

    check_build(program, device, clBuildProgram(program, 1, &device, NULL, NULL, NULL));
    cl_kernel kernel = clCreateKernel(program, "initializeCL", &err);
    check(err, "clCreateKernel");

    //Allocate memory & copy data to GPU 
    cl_mem fGPU = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * N * Q * sizeof(double), (void*)f, &err);
    check(err, "clCreateBuffer");

    cl_mem feqGPU = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * N * Q * sizeof(double), (void*)feq, &err);
    check(err, "clCreateBuffer");
    
    cl_mem f_newGPU = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * N * Q * sizeof(double), (void*)f_new, &err);
    check(err, "clCreateBuffer");

    cl_mem rhoGPU = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * N * sizeof(double), (void*)rho, &err);
    check(err, "clCreateBuffer");

    cl_mem uxGPU = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * N * sizeof(double), (void*)ux, &err);
    check(err, "clCreateBuffer");

    cl_mem uyGPU = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * N * sizeof(double), (void*)uy, &err);
    check(err, "clCreateBuffer");

    cl_mem sigmaGPU = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * N * sizeof(double), (void*)sigma, &err);
    check(err, "clCreateBuffer");

    cl_mem exGPU = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Q * sizeof(double), (void*)ex, &err);
    check(err, "clCreateBuffer");

    cl_mem eyGPU = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Q * sizeof(double), (void*)ey, &err);
    check(err, "clCreateBuffer");

    cl_mem opposGPU = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Q * sizeof(int), (void*)oppos, &err);
    check(err, "clCreateBuffer");

    cl_mem wtGPU = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Q * sizeof(double), (void*)wt, &err);
    check(err, "clCreateBuffer");

    //run kernel 
    size_t wlsize[2] = {16, 16};
    size_t wgsize[2] = {size_t(roundup(N, wlsize[0])), size_t(roundup(N, wlsize[1]))};
    printf("init parameter \n");

    CHECK(clSetKernelArg(kernel, 0, sizeof(int), (void*)&N));
    CHECK(clSetKernelArg(kernel, 1, sizeof(int), (void*)&Q));
    CHECK(clSetKernelArg(kernel, 2, sizeof(double), (void*)&DENSITY));
    CHECK(clSetKernelArg(kernel, 3, sizeof(double), (void*)&LID_VELOCITY));

    CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&exGPU));
    CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&eyGPU));
    CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&opposGPU));
    CHECK(clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&wtGPU));

    CHECK(clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&rhoGPU));
    CHECK(clSetKernelArg(kernel, 9, sizeof(cl_mem), (void*)&uxGPU));
    CHECK(clSetKernelArg(kernel, 10, sizeof(cl_mem), (void*)&uyGPU));
    CHECK(clSetKernelArg(kernel, 11, sizeof(cl_mem), (void*)&sigmaGPU));

    CHECK(clSetKernelArg(kernel, 12, sizeof(cl_mem), (void*)&fGPU));
    CHECK(clSetKernelArg(kernel, 13, sizeof(cl_mem), (void*)&feqGPU));
    CHECK(clSetKernelArg(kernel, 14, sizeof(cl_mem), (void*)&f_newGPU));

    printf("run kernel\n");

    CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, wgsize, wlsize, 0, NULL, NULL));
    CHECK(clFinish(queue));
    // Copy data back to CPU & release memory
    CHECK(clEnqueueReadBuffer(queue, fGPU, CL_TRUE, 0, N * N * Q * sizeof(double), f, 0, NULL, NULL));
    CHECK(clEnqueueReadBuffer(queue, feqGPU, CL_TRUE, 0, N * N * Q * sizeof(double), feq, 0, NULL, NULL));
    CHECK(clEnqueueReadBuffer(queue, f_newGPU, CL_TRUE, 0, N * N * Q * sizeof(double), f_new, 0, NULL, NULL));

    CHECK(clEnqueueReadBuffer(queue, rhoGPU, CL_TRUE, 0, N * N * sizeof(double), rho, 0, NULL, NULL));
    CHECK(clEnqueueReadBuffer(queue, uxGPU, CL_TRUE, 0, N * N * sizeof(double), ux, 0, NULL, NULL));
    CHECK(clEnqueueReadBuffer(queue, uyGPU, CL_TRUE, 0, N * N * sizeof(double), uy, 0, NULL, NULL));
    CHECK(clEnqueueReadBuffer(queue, sigmaGPU, CL_TRUE, 0, N * N * sizeof(double), sigma, 0, NULL, NULL));

    CHECK(clReleaseMemObject(fGPU));
    CHECK(clReleaseMemObject(feqGPU));
    CHECK(clReleaseMemObject(f_newGPU));
    CHECK(clReleaseMemObject(rhoGPU));
    CHECK(clReleaseMemObject(uxGPU));
    CHECK(clReleaseMemObject(uyGPU));
    CHECK(clReleaseMemObject(sigmaGPU));
    CHECK(clReleaseMemObject(exGPU));
    CHECK(clReleaseMemObject(eyGPU));
    CHECK(clReleaseMemObject(opposGPU));
    CHECK(clReleaseMemObject(wtGPU));
    // Release Others
    //CHECK(clReleaseProgram(program));
    CHECK(clReleaseKernel(kernel));
    //CHECK(clReleaseCommandQueue(queue));
    //CHECK(clReleaseContext(context));
}

void collideAndStreamCL(const int N, const int Q, const double DENSITY, const double LID_VELOCITY, const double REYNOLDS_NUMBER,  
                double *ex, double *ey, int *oppos, double *wt,
                double *rho, double *ux, double *uy, double* sigma, 
                double *f, double *feq, double *f_new)
{ 
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, "collideAndStreamCL", &err);
    check(err, "clCreateKernel");

    //Allocate memory & copy data to GPU 
    cl_mem fGPU = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * N * Q * sizeof(double), (void*)f, &err);
    check(err, "clCreateBuffer");

    cl_mem feqGPU = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * N * Q * sizeof(double), (void*)feq, &err);
    check(err, "clCreateBuffer");
    
    cl_mem f_newGPU = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * N * Q * sizeof(double), (void*)f_new, &err);
    check(err, "clCreateBuffer");

    cl_mem rhoGPU = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, N * N * sizeof(double), (void*)rho, &err);
    check(err, "clCreateBuffer");

    cl_mem uxGPU = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, N * N * sizeof(double), (void*)ux, &err);
    check(err, "clCreateBuffer");

    cl_mem uyGPU = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, N * N * sizeof(double), (void*)uy, &err);
    check(err, "clCreateBuffer");

    cl_mem sigmaGPU = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, N * N * sizeof(double), (void*)sigma, &err);
    check(err, "clCreateBuffer");

    cl_mem exGPU = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Q * sizeof(double), (void*)ex, &err);
    check(err, "clCreateBuffer");

    cl_mem eyGPU = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Q * sizeof(double), (void*)ey, &err);
    check(err, "clCreateBuffer");

    cl_mem opposGPU = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Q * sizeof(int), (void*)oppos, &err);
    check(err, "clCreateBuffer");

    cl_mem wtGPU = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Q * sizeof(double), (void*)wt, &err);
    check(err, "clCreateBuffer");

    //run kernel 
    size_t wlsize[2] = {16, 16};
    size_t wgsize[2] = {size_t(roundup(N, wlsize[0])), size_t(roundup(N, wlsize[1]))};

    CHECK(clSetKernelArg(kernel, 0, sizeof(int), (void*)&N));
    CHECK(clSetKernelArg(kernel, 1, sizeof(int), (void*)&Q));
    CHECK(clSetKernelArg(kernel, 2, sizeof(double), (void*)&DENSITY));
    CHECK(clSetKernelArg(kernel, 3, sizeof(double), (void*)&LID_VELOCITY));
    CHECK(clSetKernelArg(kernel, 4, sizeof(double), (void*)&REYNOLDS_NUMBER));

    CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&exGPU));
    CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&eyGPU));
    CHECK(clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&opposGPU));
    CHECK(clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&wtGPU));

    CHECK(clSetKernelArg(kernel, 9, sizeof(cl_mem), (void*)&rhoGPU));
    CHECK(clSetKernelArg(kernel, 10, sizeof(cl_mem), (void*)&uxGPU));
    CHECK(clSetKernelArg(kernel, 11, sizeof(cl_mem), (void*)&uyGPU));
    CHECK(clSetKernelArg(kernel, 12, sizeof(cl_mem), (void*)&sigmaGPU));

    CHECK(clSetKernelArg(kernel, 13, sizeof(cl_mem), (void*)&fGPU));
    CHECK(clSetKernelArg(kernel, 14, sizeof(cl_mem), (void*)&feqGPU));
    CHECK(clSetKernelArg(kernel, 15, sizeof(cl_mem), (void*)&f_newGPU));

    CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, wgsize, wlsize, 0, NULL, NULL));
    CHECK(clFinish(queue));
    
    // Copy data back to CPU & release memory
    CHECK(clEnqueueReadBuffer(queue, feqGPU, CL_TRUE, 0, N * N * Q * sizeof(double), feq, 0, NULL, NULL));
    CHECK(clEnqueueReadBuffer(queue, f_newGPU, CL_TRUE, 0, N * N * Q * sizeof(double), f_new, 0, NULL, NULL));

    CHECK(clReleaseMemObject(fGPU));
    CHECK(clReleaseMemObject(feqGPU));
    CHECK(clReleaseMemObject(f_newGPU));
    CHECK(clReleaseMemObject(rhoGPU));
    CHECK(clReleaseMemObject(uxGPU));
    CHECK(clReleaseMemObject(uyGPU));
    CHECK(clReleaseMemObject(sigmaGPU));
    CHECK(clReleaseMemObject(exGPU));
    CHECK(clReleaseMemObject(eyGPU));
    CHECK(clReleaseMemObject(opposGPU));
    CHECK(clReleaseMemObject(wtGPU));
    // Release Others
    //CHECK(clReleaseProgram(program));
    CHECK(clReleaseKernel(kernel));
    //CHECK(clReleaseCommandQueue(queue));
    //CHECK(clReleaseContext(context));
}

void macroVarCL(const int N, const int Q, const double DENSITY, const double LID_VELOCITY, const double REYNOLDS_NUMBER,  
                double *ex, double *ey, int *oppos, double *wt,
                double *rho, double *ux, double *uy, double* sigma, 
                double *f, double *feq, double *f_new)
{ 
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, "macroVarCL", &err);
    check(err, "clCreateKernel");

    //Allocate memory & copy data to GPU 
    cl_mem fGPU = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * N * Q * sizeof(double), (void*)f, &err);
    check(err, "clCreateBuffer");

    cl_mem feqGPU = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * N * Q * sizeof(double), (void*)feq, &err);
    check(err, "clCreateBuffer");
    
    cl_mem f_newGPU = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * N * Q * sizeof(double), (void*)f_new, &err);
    check(err, "clCreateBuffer");

    cl_mem rhoGPU = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * N * sizeof(double), (void*)rho, &err);
    check(err, "clCreateBuffer");

    cl_mem uxGPU = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * N * sizeof(double), (void*)ux, &err);
    check(err, "clCreateBuffer");

    cl_mem uyGPU = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * N * sizeof(double), (void*)uy, &err);
    check(err, "clCreateBuffer");

    cl_mem sigmaGPU = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * N * sizeof(double), (void*)sigma, &err);
    check(err, "clCreateBuffer");

    cl_mem exGPU = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Q * sizeof(double), (void*)ex, &err);
    check(err, "clCreateBuffer");

    cl_mem eyGPU = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Q * sizeof(double), (void*)ey, &err);
    check(err, "clCreateBuffer");

    cl_mem opposGPU = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Q * sizeof(int), (void*)oppos, &err);
    check(err, "clCreateBuffer");

    cl_mem wtGPU = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Q * sizeof(double), (void*)wt, &err);
    check(err, "clCreateBuffer");

    //run kernel 
    size_t wlsize[2] = {16, 16};
    size_t wgsize[2] = {size_t(roundup(N, wlsize[0])), size_t(roundup(N, wlsize[1]))};

    CHECK(clSetKernelArg(kernel, 0, sizeof(int), (void*)&N));
    CHECK(clSetKernelArg(kernel, 1, sizeof(int), (void*)&Q));
    CHECK(clSetKernelArg(kernel, 2, sizeof(double), (void*)&DENSITY));
    CHECK(clSetKernelArg(kernel, 3, sizeof(double), (void*)&LID_VELOCITY));
    CHECK(clSetKernelArg(kernel, 4, sizeof(double), (void*)&REYNOLDS_NUMBER));

    CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&exGPU));
    CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&eyGPU));
    CHECK(clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&opposGPU));
    CHECK(clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&wtGPU));

    CHECK(clSetKernelArg(kernel, 9, sizeof(cl_mem), (void*)&rhoGPU));
    CHECK(clSetKernelArg(kernel, 10, sizeof(cl_mem), (void*)&uxGPU));
    CHECK(clSetKernelArg(kernel, 11, sizeof(cl_mem), (void*)&uyGPU));
    CHECK(clSetKernelArg(kernel, 12, sizeof(cl_mem), (void*)&sigmaGPU));

    CHECK(clSetKernelArg(kernel, 13, sizeof(cl_mem), (void*)&fGPU));
    CHECK(clSetKernelArg(kernel, 14, sizeof(cl_mem), (void*)&feqGPU));
    CHECK(clSetKernelArg(kernel, 15, sizeof(cl_mem), (void*)&f_newGPU));

    CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, wgsize, wlsize, 0, NULL, NULL));
    CHECK(clFinish(queue));
    
    // Copy data back to CPU & release memory
    CHECK(clEnqueueReadBuffer(queue, fGPU, CL_TRUE, 0, N * N * Q * sizeof(double), f, 0, NULL, NULL));
    CHECK(clEnqueueReadBuffer(queue, rhoGPU, CL_TRUE, 0, N * N * sizeof(double), rho, 0, NULL, NULL));
    CHECK(clEnqueueReadBuffer(queue, uxGPU, CL_TRUE, 0, N * N * sizeof(double), ux, 0, NULL, NULL));
    CHECK(clEnqueueReadBuffer(queue, uyGPU, CL_TRUE, 0, N * N * sizeof(double), uy, 0, NULL, NULL));
    CHECK(clEnqueueReadBuffer(queue, sigmaGPU, CL_TRUE, 0, N * N * sizeof(double), sigma, 0, NULL, NULL));

    CHECK(clReleaseMemObject(fGPU));
    CHECK(clReleaseMemObject(feqGPU));
    CHECK(clReleaseMemObject(f_newGPU));
    CHECK(clReleaseMemObject(rhoGPU));
    CHECK(clReleaseMemObject(uxGPU));
    CHECK(clReleaseMemObject(uyGPU));
    CHECK(clReleaseMemObject(sigmaGPU));
    CHECK(clReleaseMemObject(exGPU));
    CHECK(clReleaseMemObject(eyGPU));
    CHECK(clReleaseMemObject(opposGPU));
    CHECK(clReleaseMemObject(wtGPU));
    // Release Others
    //CHECK(clReleaseProgram(program));
    CHECK(clReleaseKernel(kernel));
    //CHECK(clReleaseCommandQueue(queue));
    //CHECK(clReleaseContext(context));
}

void LatticeBoltzmann(const int N, const int Q, const double DENSITY, const double LID_VELOCITY, const double REYNOLDS_NUMBER,  
                double *ex, double *ey, int *oppos, double *wt,
                double *rho, double *ux, double *uy, double* sigma, 
                double *f, double *feq, double *f_new)
{
    cl_int err;
    cl_kernel kernelCollideStream = clCreateKernel(program, "collideAndStreamCL", &err);
    check(err, "clCreateKernel");

    cl_kernel kernelUpdate = clCreateKernel(program, "macroVarCL", &err);
    check(err, "clCreateKernel");

    //Allocate memory & copy data to GPU 
    cl_mem fGPU = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * N * Q * sizeof(double), (void*)f, &err);
    check(err, "clCreateBuffer");

    cl_mem feqGPU = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * N * Q * sizeof(double), (void*)feq, &err);
    check(err, "clCreateBuffer");
    
    cl_mem f_newGPU = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * N * Q * sizeof(double), (void*)f_new, &err);
    check(err, "clCreateBuffer");

    cl_mem rhoGPU = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * N * sizeof(double), (void*)rho, &err);
    check(err, "clCreateBuffer");

    cl_mem uxGPU = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * N * sizeof(double), (void*)ux, &err);
    check(err, "clCreateBuffer");

    cl_mem uyGPU = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * N * sizeof(double), (void*)uy, &err);
    check(err, "clCreateBuffer");

    cl_mem sigmaGPU = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, N * N * sizeof(double), (void*)sigma, &err);
    check(err, "clCreateBuffer");

    cl_mem exGPU = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Q * sizeof(double), (void*)ex, &err);
    check(err, "clCreateBuffer");

    cl_mem eyGPU = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Q * sizeof(double), (void*)ey, &err);
    check(err, "clCreateBuffer");

    cl_mem opposGPU = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Q * sizeof(int), (void*)oppos, &err);
    check(err, "clCreateBuffer");

    cl_mem wtGPU = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Q * sizeof(double), (void*)wt, &err);
    check(err, "clCreateBuffer");

    //run kernel 
    size_t wlsize[2] = {16, 16};
    size_t wgsize[2] = {size_t(roundup(N, wlsize[0])), size_t(roundup(N, wlsize[1]))};

    CHECK(clSetKernelArg(kernelCollideStream, 0, sizeof(int), (void*)&N));
    CHECK(clSetKernelArg(kernelCollideStream, 1, sizeof(int), (void*)&Q));
    CHECK(clSetKernelArg(kernelCollideStream, 2, sizeof(double), (void*)&DENSITY));
    CHECK(clSetKernelArg(kernelCollideStream, 3, sizeof(double), (void*)&LID_VELOCITY));
    CHECK(clSetKernelArg(kernelCollideStream, 4, sizeof(double), (void*)&REYNOLDS_NUMBER));

    CHECK(clSetKernelArg(kernelCollideStream, 5, sizeof(cl_mem), (void*)&exGPU));
    CHECK(clSetKernelArg(kernelCollideStream, 6, sizeof(cl_mem), (void*)&eyGPU));
    CHECK(clSetKernelArg(kernelCollideStream, 7, sizeof(cl_mem), (void*)&opposGPU));
    CHECK(clSetKernelArg(kernelCollideStream, 8, sizeof(cl_mem), (void*)&wtGPU));

    CHECK(clSetKernelArg(kernelCollideStream, 9, sizeof(cl_mem), (void*)&rhoGPU));
    CHECK(clSetKernelArg(kernelCollideStream, 10, sizeof(cl_mem), (void*)&uxGPU));
    CHECK(clSetKernelArg(kernelCollideStream, 11, sizeof(cl_mem), (void*)&uyGPU));
    CHECK(clSetKernelArg(kernelCollideStream, 12, sizeof(cl_mem), (void*)&sigmaGPU));

    CHECK(clSetKernelArg(kernelCollideStream, 13, sizeof(cl_mem), (void*)&fGPU));
    CHECK(clSetKernelArg(kernelCollideStream, 14, sizeof(cl_mem), (void*)&feqGPU));
    CHECK(clSetKernelArg(kernelCollideStream, 15, sizeof(cl_mem), (void*)&f_newGPU));

    CHECK(clEnqueueNDRangeKernel(queue, kernelCollideStream, 2, NULL, wgsize, wlsize, 0, NULL, NULL));
    CHECK(clFinish(queue));

    CHECK(clSetKernelArg(kernelUpdate, 0, sizeof(int), (void*)&N));
    CHECK(clSetKernelArg(kernelUpdate, 1, sizeof(int), (void*)&Q));
    CHECK(clSetKernelArg(kernelUpdate, 2, sizeof(double), (void*)&DENSITY));
    CHECK(clSetKernelArg(kernelUpdate, 3, sizeof(double), (void*)&LID_VELOCITY));
    CHECK(clSetKernelArg(kernelUpdate, 4, sizeof(double), (void*)&REYNOLDS_NUMBER));

    CHECK(clSetKernelArg(kernelUpdate, 5, sizeof(cl_mem), (void*)&exGPU));
    CHECK(clSetKernelArg(kernelUpdate, 6, sizeof(cl_mem), (void*)&eyGPU));
    CHECK(clSetKernelArg(kernelUpdate, 7, sizeof(cl_mem), (void*)&opposGPU));
    CHECK(clSetKernelArg(kernelUpdate, 8, sizeof(cl_mem), (void*)&wtGPU));

    CHECK(clSetKernelArg(kernelUpdate, 9, sizeof(cl_mem), (void*)&rhoGPU));
    CHECK(clSetKernelArg(kernelUpdate, 10, sizeof(cl_mem), (void*)&uxGPU));
    CHECK(clSetKernelArg(kernelUpdate, 11, sizeof(cl_mem), (void*)&uyGPU));
    CHECK(clSetKernelArg(kernelUpdate, 12, sizeof(cl_mem), (void*)&sigmaGPU));

    CHECK(clSetKernelArg(kernelUpdate, 13, sizeof(cl_mem), (void*)&fGPU));
    CHECK(clSetKernelArg(kernelUpdate, 14, sizeof(cl_mem), (void*)&feqGPU));
    CHECK(clSetKernelArg(kernelUpdate, 15, sizeof(cl_mem), (void*)&f_newGPU));

    CHECK(clEnqueueNDRangeKernel(queue, kernelUpdate, 2, NULL, wgsize, wlsize, 0, NULL, NULL));
    CHECK(clFinish(queue));
    
    // Copy data back to CPU & release memory
    CHECK(clEnqueueReadBuffer(queue, feqGPU, CL_TRUE, 0, N * N * Q * sizeof(double), feq, 0, NULL, NULL));
    CHECK(clEnqueueReadBuffer(queue, f_newGPU, CL_TRUE, 0, N * N * Q * sizeof(double), f_new, 0, NULL, NULL));
    CHECK(clEnqueueReadBuffer(queue, fGPU, CL_TRUE, 0, N * N * Q * sizeof(double), f, 0, NULL, NULL));
    CHECK(clEnqueueReadBuffer(queue, rhoGPU, CL_TRUE, 0, N * N * sizeof(double), rho, 0, NULL, NULL));
    CHECK(clEnqueueReadBuffer(queue, uxGPU, CL_TRUE, 0, N * N * sizeof(double), ux, 0, NULL, NULL));
    CHECK(clEnqueueReadBuffer(queue, uyGPU, CL_TRUE, 0, N * N * sizeof(double), uy, 0, NULL, NULL));
    CHECK(clEnqueueReadBuffer(queue, sigmaGPU, CL_TRUE, 0, N * N * sizeof(double), sigma, 0, NULL, NULL));

    CHECK(clReleaseMemObject(fGPU));
    CHECK(clReleaseMemObject(feqGPU));
    CHECK(clReleaseMemObject(f_newGPU));
    CHECK(clReleaseMemObject(rhoGPU));
    CHECK(clReleaseMemObject(uxGPU));
    CHECK(clReleaseMemObject(uyGPU));
    CHECK(clReleaseMemObject(sigmaGPU));
    CHECK(clReleaseMemObject(exGPU));
    CHECK(clReleaseMemObject(eyGPU));
    CHECK(clReleaseMemObject(opposGPU));
    CHECK(clReleaseMemObject(wtGPU));
    
    // Release Others
    //CHECK(clReleaseProgram(program));
    CHECK(clReleaseKernel(kernelCollideStream));
    CHECK(clReleaseKernel(kernelUpdate));
    //CHECK(clReleaseCommandQueue(queue));
    //CHECK(clReleaseContext(context));
}

void ReleaseRessource()
{
    CHECK(clReleaseProgram(program));
    CHECK(clReleaseCommandQueue(queue));
    CHECK(clReleaseContext(context));
}
