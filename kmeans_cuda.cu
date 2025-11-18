#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h> //win io.h
#include <float.h>

#define MAX_NCLUST 20

struct KMData {
    int ndata;
    int dim;
    float* features;
    int* assigns;
    int* labels;
    int nlabels;
};

struct KMClust {
    int nclust;
    int dim;
    float* features;
    int* counts;
};

struct KMData* kmdata_load(char *filename);

struct KMClust* kmclust_new(int nclust, int dim);

int filestats(char *filename, ssize_t *tot_tokens, ssize_t *tot_lines);

void save_pgm_files(struct KMClust* clust, char* savedir);

__global__ void calculate_clusters(int ndata, int dim, int nclust, float *data_features, float *clust_features, int *assigns, int *counts)
{   
    long i = threadIdx.x + blockDim.x * blockIdx.x;

    // calculate cluster features at a specific features dimension and cluster

    if(i < dim*nclust){
        long myd = i % dim; // feature dimension
        long myc = i / dim; // cluster index
        
        for(int j=0; j<ndata; j++){       //sum up data in each cluster
            int c = assigns[j];
            if(myc == c){
                clust_features[myc*dim + myd] += data_features[j*dim + myd];
            }
        }

        // divide by ndatas of data to get mean of cluster center
        if(counts[myc] > 0){
          clust_features[myc*dim + myd] = clust_features[myc*dim + myd] / counts[myc];
        }
    }
}

__global__ void clusters_assignment(int ndata, int dim, int nclust, float *data_features, float *centers, int *assigns, int *counts, int *nchanges)
{   

    long i = threadIdx.x + blockDim.x * blockIdx.x;

    // calculate the cluster assignment of data i

    if(i<ndata){

        int best_clust = -1;
        float best_distsq = FLT_MAX;

        for(int c=0; c<nclust; c++){
            float distsq =0.0;
            for(int d=0; d<dim; d++){
                float diff = data_features[i*dim+d] - centers[c*dim+d];
                distsq += diff*diff;
            }
            if(distsq < best_distsq){
                best_clust = c;
                best_distsq = distsq;
            }
        }
        
        // use atomicAdd to avoid memory contention
        atomicAdd(&counts[best_clust], 1);
        if(best_clust != assigns[i]){
            atomicAdd(nchanges, 1);
            assigns[i] = best_clust;
        }

    }
    
    
}

int main( int argc, char *argv[] )
{   
    if(argc < 3){
      printf("usage: program_name <datafile> <nclust> [savedir] [maxiter]");
      exit(1);
    }

    char* datafile = argv[1];
    int nclust = atoi(argv[2]);
    char* savedir = ".";
    int MAXITER = 100;

    if(argc > 3){
      savedir = argv[3];
      mkdir(savedir, S_IRUSR | S_IWUSR | S_IXUSR);
    }

    if(argc > 4){
      MAXITER = atoi(argv[4]);
    }

    printf("datafile: %s\n",datafile);
    printf("nclust: %d\n",nclust);
    printf("savedir: %s\n",savedir);

    struct KMData* data = kmdata_load(datafile);        //read in the data file, allocate cluster space
    struct KMClust* clust = kmclust_new(nclust, data->dim);

    printf("ndata: %d\n", data->ndata);
    printf("dim: %d\n",data->dim);
    printf("\n");

    for(int i=0; i<data->ndata; i++){       //random, regular initial cluster assignment
      int c = i % clust->nclust;
      data->assigns[i]=c;
    }

    for(int c=0; c<clust->nclust; c++){
      int icount = data->ndata/clust->nclust;
      int extra = 0;
      if(c<data->ndata%clust->nclust){
        extra = 1;
      }
      clust->counts[c] = icount + extra;
    }

    // KMEANS MAIN ALGORITHM PART

    int curiter = 1;        //current iteration
    int nchanges = data->ndata;       //check for changes in cluster assignment; 0 is converged
    printf("==CLUSTERING: MAXITER %d==\n", MAXITER);
    printf("ITER NCHANGE CLUST_COUNTS\n");

    // define device variables
    float *dev_data_features;
    cudaMalloc((void**) &dev_data_features, data->ndata*data->dim*sizeof(float));
    cudaMemcpy(dev_data_features, data->features, data->ndata*data->dim*sizeof(float), cudaMemcpyHostToDevice);

    float *dev_clust_features;
    cudaMalloc((void**) &dev_clust_features, clust->nclust*clust->dim*sizeof(float));

    int *dev_assigns;
    cudaMalloc((void**) &dev_assigns, data->ndata*sizeof(int));
    cudaMemcpy(dev_assigns, data->assigns, data->ndata*sizeof(int), cudaMemcpyHostToDevice);

    int *dev_nchanges;
    cudaMalloc((void**) &dev_nchanges, sizeof(int));

    int *dev_counts;
    cudaMalloc((void**) &dev_counts, clust->nclust*sizeof(int));
    cudaMemcpy(dev_counts, clust->counts, clust->nclust*sizeof(int), cudaMemcpyHostToDevice);

    while(nchanges>0 && curiter<=MAXITER){        //loop until converges

      //DETERMINE NEW CLUSTER CENTERS

      long nthreads = 256;
      long nblocks = (data->dim*clust->nclust + (nthreads - 1)) / nthreads;

      // update device variable
      cudaMemset(dev_clust_features, 0, clust->nclust*clust->dim*sizeof(float));

      // CUDA parallel code
      calculate_clusters<<<nblocks, nthreads>>>(data->ndata, data->dim, clust->nclust, dev_data_features, dev_clust_features, dev_assigns, dev_counts);
    
      // DETERMINE NEW CLUSTER ASSIGNMENTS FOR EACH DATA

      nthreads = 256;
      nblocks = (data->ndata + (nthreads - 1)) / nthreads;

      // update device variable
      cudaMemset(dev_nchanges, 0, sizeof(int));
      cudaMemset(dev_counts, 0, clust->nclust*sizeof(int));
      
      // CUDA parallel code
      clusters_assignment<<<nblocks, nthreads>>>(data->ndata, data->dim, clust->nclust, dev_data_features, dev_clust_features, dev_assigns, dev_counts, dev_nchanges);

      // copy device memory to host
      cudaMemcpy(&nchanges, dev_nchanges, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(clust->counts, dev_counts, clust->nclust*sizeof(int), cudaMemcpyDeviceToHost);

      printf("%3d: %5d |", curiter, nchanges);
      for(int c=0; c<nclust; c++){
        printf(" %4d", clust->counts[c]);
      }
      printf("\n");
      curiter += 1;
    }

    // copy device memory to host
    cudaMemcpy(data->assigns, dev_assigns, data->ndata*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(clust->features, dev_clust_features, clust->nclust*clust->dim*sizeof(float), cudaMemcpyDeviceToHost);

    if(curiter > MAXITER)
      printf("WARNING: maximum iteration %d exceeded, may not have conveged\n", MAXITER);
    else
      printf("CONVERGED: after %d iterations\n", curiter);
    printf("\n");




    // CLEANUP + OUTPUT

    // CONFUSION MATRIX

    int* confusion = (int*) malloc(data->nlabels*nclust * sizeof(int));        //confusion matrix: labels * clusters big
    for(int i=0; i<data->nlabels*nclust; i++){
      confusion[i] = 0;
    }

    for(int i=0; i<data->ndata; i++){       //count which labels in which clusters
      int label = data->labels[i];
      int assign = data->assigns[i];
      confusion[label*nclust + assign] += 1;
    }

    printf("==CONFUSION MATRIX + COUNTS==\n");
    printf("LABEL \\ CLUST\n");

    printf("%3s","");       // confusion matrix header
    for(int j=0; j<nclust;j++){
      printf(" %4d", j);
    }
    printf(" %4s\n", "TOT");

    for(int i=0; i<data->nlabels; i++){       //each row of confusion matrix
      printf("%2d:", i);
      int tot = 0;
      for(int j=0; j<nclust; j++){
        printf(" %4d", confusion[i*nclust+j]);
        tot += confusion[i*nclust+j];
      }
      printf(" %4d\n", tot);
    }


    printf("TOT");        //final total row of confusion matrix
    int tot = 0;
    for(int c=0; c<nclust; c++){
      printf(" %4d", clust->counts[c]);
      tot += clust->counts[c];
    }

    printf(" %4d\n", tot);
    printf("\n");

    // LABEL FILE OUTPUT

    char outfile[200];
    sprintf(outfile, "%s/labels.txt", savedir);

    printf("Saving cluster labels to file %s/labels.txt\n", savedir);

    FILE* fout;
    fout = fopen(outfile, "w+");

    for(int i=0; i<data->ndata; i++){
      fprintf(fout, "%2d %2d\n", data->labels[i], data->assigns[i]);
    }
    fclose(fout);

    save_pgm_files(clust, savedir);

    free(data->features);
    free(data->labels);
    free(data->assigns);
    free(data);

    free(clust->counts);
    free(clust->features);
    free(clust);

    free(confusion);

    cudaFree(dev_data_features);
    cudaFree(dev_clust_features);
    cudaFree(dev_assigns);
    cudaFree(dev_nchanges);
    cudaFree(dev_counts);

    return 0;
}



struct KMData* kmdata_load(char *filename)
{
    ssize_t tot_tokens, tot_lines;
    filestats(filename, &tot_tokens, &tot_lines);

    FILE *fin = fopen(filename,"r");
    struct KMData * kmdata = (struct KMData*) malloc(sizeof(struct KMData));
    kmdata->ndata = tot_lines;
    kmdata->dim = tot_tokens/tot_lines - 2;

    kmdata->features = (float*) malloc(kmdata->ndata*kmdata->dim * sizeof(float));
    kmdata->labels = (int*) malloc(kmdata->ndata*sizeof(int));
    
    int max_label = 0;
    for(int l=0; l<kmdata->ndata; l++){
        fscanf(fin, "%d :", &kmdata->labels[l]);
        if(kmdata->labels[l] > max_label){
          max_label = kmdata->labels[l];
        }
        for(int i=0; i<kmdata->dim; i++){
            fscanf(fin, "%f", &kmdata->features[l*kmdata->dim+i]);
        }
        //printf("\n");
    }
    fclose(fin);

    kmdata->nlabels = max_label + 1;

    kmdata->assigns = (int*) malloc(kmdata->ndata*sizeof(int));
    return kmdata;
};


struct KMClust* kmclust_new(int nclust, int dim){
  struct KMClust* clust = (struct KMClust*) malloc(sizeof(struct KMClust));
  clust->nclust = nclust;
  clust->dim = dim;
  clust->features = (float*) malloc(nclust*dim*sizeof(float));
  clust->counts = (int*) malloc(nclust*sizeof(int));

  for(int i=0; i<nclust; i++){
    clust->counts[i]=0;
    for(int j=0; j<dim; j++){
      clust->features[dim*i+j]=0;
    }
  }
  return clust;
}

void save_pgm_files(struct KMClust* clust, char* savedir){
  int dim_root = (int)sqrt(clust->dim);
  if(clust->dim % dim_root == 0){
    printf("Saving cluster centers to %s/cent_0000.pgm ...\n", savedir);
    float maxfeat = 0;
    for(int i=0; i<clust->dim*clust->nclust; i++){
      if(clust->features[i] > maxfeat){
        maxfeat = clust->features[i];
      }
    }
    // write different PGM for each cluster
    for(int c=0; c<clust->nclust; c++){ 
      char outfile[200];
      sprintf(outfile, "%s/cent_%04d.pgm", savedir, c);

      FILE *fout;
      fout = fopen(outfile,"w");
      // start writing PGM files
      fprintf(fout, "P2\n");
      fprintf(fout, "%d %d\n", dim_root, dim_root);
      fprintf(fout, "%.0f\n", maxfeat);

      for(int d=0; d<clust->dim; d++){
        if(d>0 && d%dim_root==0){
          fprintf(fout, "\n");
        }
        fprintf(fout, "%3.0f ", clust->features[c*clust->dim+d]);
      }
      fprintf(fout,"\n");

      fclose(fout);
    }


  }
}
