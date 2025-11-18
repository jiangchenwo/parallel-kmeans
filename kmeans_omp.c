#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <float.h>
#include <omp.h>

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

    while(nchanges>0 && curiter<=MAXITER){        //loop until converges

      //DETERMINE NEW CLUSTER CENTERS
      for(int c=0; c<clust->nclust;c++){        //reset cluster centers to 0.0
        for(int d=0; d<clust->dim; d++){
          clust->features[c*clust->dim + d] = 0.0;
        }
      }

      #pragma omp parallel
      {
        // Define local thread features
        float thread_features[clust->nclust*clust->dim];
        for(int c=0; c<clust->nclust;c++){
          for(int d=0; d<clust->dim; d++){
            thread_features[c*clust->dim + d] = 0.0;
          }
        }

        #pragma omp for
        for(int i=0; i<data->ndata; i++){       //sum up data in each cluster
          int c = data->assigns[i];
          for(int d=0; d<clust->dim; d++){
            thread_features[c*clust->dim + d] += data->features[i*data->dim + d];
          }
        }

        // Reduction on thread features
        for(int c=0; c<clust->nclust; c++){
          for(int d=0; d<clust->dim; d++){ //divide by ndatas of data to get mean of cluster center
            #pragma omp atomic
            clust->features[c*clust->dim + d] += thread_features[c*clust->dim + d];
          }
        }
      }

      for(int c=0; c<clust->nclust; c++){       //divide by ndatas of data to get mean of cluster center
        if(clust->counts[c] > 0){
          for(int d=0; d<clust->dim; d++){
            clust->features[c*clust->dim + d] = clust->features[c*clust->dim + d] / clust->counts[c];
          }
        }
      }

      // DETERMINE NEW CLUSTER ASSIGNMENTS FOR EACH DATA
      for(int c=0; c<clust->nclust; c++){       // reset cluster counts to 0
        clust->counts[c] = 0;               //re-init here to support first iteration
      }

      nchanges = 0;
      #pragma omp parallel
      {
        int thread_clust_counts[clust->nclust]; // local thread variable for cluster counts 
        for(int c=0; c<clust->nclust; c++){       // initialization
          thread_clust_counts[c] = 0;          
        }

        int best_clust;
        float distsq, diff, best_distsq;

        #pragma omp for reduction(+: nchanges)
        for(int i=0; i<data->ndata; i++){       //iterate over all data
          best_clust = -1;
          best_distsq = FLT_MAX;

          for(int c=0; c<clust->nclust; c++){       //compare data to each cluster and assign to closest
            distsq = 0.0;
            for(int d=0; d<clust->dim; d++){
              diff = data->features[i*data->dim + d] - clust->features[c*clust->dim + d];
              distsq += diff*diff;
            }
            if(distsq < best_distsq){
              best_clust = c;
              best_distsq = distsq;
            }
          }

          thread_clust_counts[best_clust] += 1; // update local cluster counts
          if(best_clust != data->assigns[i]){
            nchanges += 1;
            data->assigns[i] = best_clust;
          }
        }

        for(int c=0; c<clust->nclust; c++){ // reduction on cluster counts
          #pragma omp atomic
          clust->counts[c] += thread_clust_counts[c];
        }
      }
      

      printf("%3d: %5d |", curiter, nchanges);
      for(int c=0; c<clust->nclust; c++){
        printf(" %4d", clust->counts[c]);
      }
      printf("\n");
      curiter += 1;
    }

    if(curiter > MAXITER)
      printf("WARNING: maximum iteration %d exceeded, may not have conveged\n", MAXITER);
    else
      printf("CONVERGED: after %d iterations\n", curiter);
    printf("\n");




    // CLEANUP + OUTPUT

    // CONFUSION MATRIX

    int* confusion = malloc(data->nlabels*nclust * sizeof(int));        //confusion matrix: labels * clusters big
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




    return 0;
}



struct KMData* kmdata_load(char *filename)
{
    ssize_t tot_tokens, tot_lines;
    filestats(filename, &tot_tokens, &tot_lines);

    FILE *fin = fopen(filename,"r");
    struct KMData * kmdata = malloc(sizeof(struct KMData));
    kmdata->ndata = tot_lines;
    kmdata->dim = tot_tokens/tot_lines - 2;

    kmdata->features = malloc(kmdata->ndata*kmdata->dim * sizeof(float));
    kmdata->labels = malloc(kmdata->ndata*sizeof(int));
    
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

    kmdata->assigns = malloc(kmdata->ndata*sizeof(int));
    return kmdata;
};


struct KMClust* kmclust_new(int nclust, int dim){
  struct KMClust* clust = malloc(sizeof(struct KMClust));
  clust->nclust = nclust;
  clust->dim = dim;
  clust->features = malloc(nclust*dim*sizeof(float));
  clust->counts = malloc(nclust*sizeof(int));

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

