#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

/*
 *   this is a classifier for the MNIST dataset:
 *       http://yann.lecun.com/exdb/mnist/
 *
 *   its structure is:
 *     - 28x28 pixel input
 *     - 3x3 convolutional layer, 6 filters
 *     - reLU
 *     - fully connected layer
 *     - softmax
 *   see `forward` below for the calculation
 *
 *   Sample Results
 *   --------------
 *
 *
 *
 *   TODO:
 *     - understand learning parameters
 *     - display filters learned
 *     - second convolution layer?
 *     - second fully connected?
 *     - code generation
 */

#define L1DEPTH 6
#define WINDOW 3
#define LABELS 10
#define INPUTWIDTH 28
#define DROPOUTWIDTH 28
#define EXAMPLES 8000 // number of tests
#define TESTS 1000

// buffers for training data
double data[EXAMPLES][INPUTWIDTH][INPUTWIDTH];
int labels[EXAMPLES];
double test_data[TESTS][INPUTWIDTH][INPUTWIDTH];
int test_labels[TESTS];

// parameters/layer values from forward pass
double f1[L1DEPTH][WINDOW][WINDOW];
double full[LABELS][L1DEPTH][DROPOUTWIDTH][DROPOUTWIDTH];
double bias[LABELS];
double input[INPUTWIDTH+2][INPUTWIDTH+2];
double L1[L1DEPTH][INPUTWIDTH][INPUTWIDTH];
double dropout[L1DEPTH][DROPOUTWIDTH][DROPOUTWIDTH];
double softmax[LABELS];

// parameter/layer gradients
double _f1[L1DEPTH][WINDOW][WINDOW];
double _full[LABELS][L1DEPTH][DROPOUTWIDTH][DROPOUTWIDTH];
double _bias[LABELS];
double _input[INPUTWIDTH+2][INPUTWIDTH+2];
double _L1[L1DEPTH][INPUTWIDTH][INPUTWIDTH];
double _dropout[L1DEPTH][DROPOUTWIDTH][DROPOUTWIDTH];
double _softmax[LABELS];

// stuff
double softmaxNormalizer;
double softmaxDistribution[LABELS];
int maxScore;
int countRight[LABELS];
int countCases[LABELS];

void printDropout()
{
  for(int r = 0; r < DROPOUTWIDTH; r++) {
  for(int c = 0; c < DROPOUTWIDTH; c++) {
    printf("%0.1f, ", dropout[0][r][c]);
  }
  printf("\n");
  }
}

void printInput()
{
  for(int r = 0; r < INPUTWIDTH+2; r++) {
    for(int c = 0; c < INPUTWIDTH+2; c++) {
      printf("%0.1f, ", input[r][c]);
    }
    printf("\n");
  }
}
void printSoftmax()
{
  for (int out = 0; out < LABELS; out++)
    printf("%0.1f, ", softmax[out]);
  //printf("\n");
  //for (int out = 0; out < LABELS; out++)
  //  printf("%0.1f, ", _softmax[out]);
  printf("\n");
}
void printThetaNorm()
{
  double biasNorm = 0;
  for (int out = 0; out < LABELS; out++) {
    biasNorm += bias[out]*bias[out];
  for (int d = 0; d < L1DEPTH; d++) {
    double norm = 0;
    for (int r = 0; r < DROPOUTWIDTH; r++) {
    for (int c = 0; c < DROPOUTWIDTH; c++) {
      norm += full[out][d][r][c] * full[out][d][r][c];
    }}
    printf("%d: %0.2f, ", out, norm);
  }}
  printf("\nbias: %0.2f\n", biasNorm);
}

double randun()
{
  return ((double)rand())/((double)RAND_MAX)*2.0-1.0;
}

void forward()
{
  // TODO layers before dropout
  for (int d = 0; d < L1DEPTH; d++) {
  for (int r = 0; r < DROPOUTWIDTH; r++) {
  for (int c = 0; c < DROPOUTWIDTH; c++) {
    L1[d][r][c] = 0.0;
    for(int wr = 0; wr < WINDOW; wr++) {
      for(int wc = 0; wc < WINDOW; wc++) {
        L1[d][r][c] += f1[d][wr][wc] * input[r+wr][c+wc];
      }
    }
  }}}
  for (int d = 0; d < L1DEPTH; d++) {
  for (int r = 0; r < DROPOUTWIDTH; r++) {
  for (int c = 0; c < DROPOUTWIDTH; c++) {
    // TODO actually dropout
    dropout[d][r][c] = L1[d][r][c] > 0 ? L1[d][r][c] : 0;
  }}}

  // compute dot product between full[out] and dropout into softmax[out]
  for (int out = 0; out < LABELS; out++) {
    softmax[out] = bias[out];
  }
  for (int out = 0; out < LABELS; out++) {
  for (int d = 0; d < L1DEPTH; d++) {
  for (int r = 0; r < DROPOUTWIDTH; r++) {
  for (int c = 0; c < DROPOUTWIDTH; c++) {
    softmax[out] += full[out][d][r][c] * dropout[d][r][c];
  }}}}

  // normalize and store maximum score index
  double maxv = softmax[0];
  maxScore = 0;
  for (int out = 0; out < LABELS; out++)
    if (softmax[out] > maxv) {
      maxv = softmax[out];
      maxScore = out;
    }
  for (int out = 0; out < LABELS; out++)
    softmax[out] -= maxv;

  // store label distribution
  softmaxNormalizer = 0;
  for (int out = 0; out < LABELS; out++)
    softmaxNormalizer += exp(softmax[out]);

  for (int out = 0; out < LABELS; out++)
    softmaxDistribution[out] = exp(softmax[out]) / softmaxNormalizer;
}

void backward(int label)
{
  // compute gradient at softmax score
  for (int out = 0; out < LABELS; out++)
    _softmax[out] = -softmaxDistribution[out];

  _softmax[label] += 1;

  // TODO factor out hyperparameters
  // gradient: fully connected layer
  for (int d = 0; d < L1DEPTH; d++) {
  for (int r = 0; r < DROPOUTWIDTH; r++) {
  for (int c = 0; c < DROPOUTWIDTH; c++) {
    _dropout[d][r][c] = 0.0;
  }}}
  for (int out = 0; out < LABELS; out++) {
    _bias[out] = _softmax[out];
    for (int d = 0; d < L1DEPTH; d++) {
    for (int r = 0; r < DROPOUTWIDTH; r++) {
    for (int c = 0; c < DROPOUTWIDTH; c++) {
      _full[out][d][r][c] = _softmax[out] * dropout[d][r][c];
      _dropout[d][r][c] += _softmax[out] * full[out][d][r][c];
    }}}
  }
  // gradient: max(0,convolution output L1)
  for (int d = 0; d < L1DEPTH; d++) {
  for (int r = 0; r < DROPOUTWIDTH; r++) {
  for (int c = 0; c < DROPOUTWIDTH; c++) {
    _L1[d][r][c] = L1[d][r][c] > 0 ? _dropout[d][r][c] : 0;
  }}}
  // gradient: convolution filters
  for (int d = 0; d < L1DEPTH; d++) {
    for(int wr = 0; wr < WINDOW; wr++) {
      for(int wc = 0; wc < WINDOW; wc++) {
        _f1[d][wr][wc] = 0;
      }}}
  for (int r = 0; r < DROPOUTWIDTH; r++) {
  for (int c = 0; c < DROPOUTWIDTH; c++) {
  for (int d = 0; d < L1DEPTH; d++) {
    for(int wr = 0; wr < WINDOW; wr++) {
      for(int wc = 0; wc < WINDOW; wc++) {
        _f1[d][wr][wc] += _L1[d][r][c] * input[r+wr][c+wc];
      }}
  }}}

  // apply gradients
  double alpha = 1.0;
  for (int out = 0; out < LABELS; out++) {
    bias[out] *= 0.999999;
    bias[out] += alpha * _bias[out] / 3.3;
  }
  for (int d = 0; d < L1DEPTH; d++) {
  for (int r = 0; r < DROPOUTWIDTH; r++) {
  for (int c = 0; c < DROPOUTWIDTH; c++) {
  for (int out = 0; out < LABELS; out++) {
      full[out][d][r][c] *= 0.999999;
      full[out][d][r][c] += alpha * _full[out][d][r][c] / (28.0*3.3);
  }}}}
  for (int d = 0; d < L1DEPTH; d++) {
    for(int wr = 0; wr < WINDOW; wr++) {
      for(int wc = 0; wc < WINDOW; wc++) {
        f1[d][wr][wc] *= 0.999999;
        f1[d][wr][wc] += alpha * _f1[d][wr][wc]/50;
      }}}
}

void init()
{
  // bias, softmax weights
  for (int out = 0; out < LABELS; out++) {
    bias[out] = 0.0;
    for (int d = 0; d < L1DEPTH; d++) {
      for (int r = 0; r < DROPOUTWIDTH; r++) {
        for (int c = 0; c < DROPOUTWIDTH; c++) {
          full[out][d][r][c] = randun();
        }
      }
    }
  }
  // convolution filters
  for (int d = 0; d < L1DEPTH; d++) {
    for(int wr = 0; wr < WINDOW; wr++) {
      for(int wc = 0; wc < WINDOW; wc++) {
        // TODO add positive bias?
        f1[d][wr][wc] = randun();
      }
    }
  }
  // input padding
  for (int i = 0; i < INPUTWIDTH+1; i++) {
    input[0][i] = 0.0;
    input[i][INPUTWIDTH-1] = 0.0;
    input[INPUTWIDTH-1][i+1];
    input[i+1][0] = 0.0;
  }
}

bool read()
{
  FILE *fp;
  char train_data_file[] = "train-images-idx3-ubyte";
  char train_labels_file[] = "train-labels-idx1-ubyte";
  char test_data_file[] = "t10k-images-idx3-ubyte";
  char test_labels_file[] = "t10k-labels-idx1-ubyte";
  char skip[16];
  char buffer[INPUTWIDTH*INPUTWIDTH];

  /* Load training data */
  fp = fopen(train_data_file, "r");
  if (fp == NULL) {
    printf("fail: %s\n", train_data_file);
    return true;
  }
  fread(skip, 1, 16, fp); // skip header
  for(int i = 0; i < EXAMPLES; i++) {
    fgets(buffer, INPUTWIDTH*INPUTWIDTH, fp);
    for (int r = 0; r < INPUTWIDTH; r++) {
      for (int c = 0; c < INPUTWIDTH; c++) {
        data[i][r][c] = ((double)(buffer[r*INPUTWIDTH+c]-128))/255.0;
      }
    }
  }
  fclose(fp);
  fp = fopen(train_labels_file, "r");
  if (fp == NULL) {
    printf("fail: %s\n", train_labels_file);
    return true;
  }
  fread(skip, 1, 8, fp); // skip header
  for(int i = 0; i < EXAMPLES; i++) {
    labels[i] = fgetc(fp);
  }
  fclose(fp);
  /* Load testing data */
  fp = fopen(test_data_file, "r");
  if (fp == NULL) {
    printf("fail: %s\n", test_data_file);
    return true;
  }
  fread(skip, 1, 16, fp); // skip header
  for(int i = 0; i < TESTS; i++) {
    fgets(buffer, INPUTWIDTH*INPUTWIDTH, fp);
    for (int r = 0; r < INPUTWIDTH; r++) {
      for (int c = 0; c < INPUTWIDTH; c++) {
        test_data[i][r][c] = ((double)(buffer[r*INPUTWIDTH+c]-128))/255.0;
      }
    }
  }
  fclose(fp);
  fp = fopen(test_labels_file, "r");
  if (fp == NULL) {
    printf("fail: %s\n", test_labels_file);
    return true;
  }
  fread(skip, 1, 8, fp); // skip header
  for(int i = 0; i < TESTS; i++) {
    test_labels[i] = fgetc(fp);
  }
  fclose(fp);
}

void train()
{
  for (int epoch = 0; epoch < 15; epoch++) {
    printf("===== EPOCH: %d =====\n", epoch);

    // record improvements each epoch
    for (int i = 0; i < LABELS; i++) {
      countRight[i] = 0;
      countCases[i] = 0;
    }

    // Train on one image at a time
    for(int index = 0; index < EXAMPLES; index++) {
      // load an image
      for (int r = 0; r < INPUTWIDTH; r++) {
        for (int c = 0; c < INPUTWIDTH; c++) {
          // 1-padding offset
          input[r+1][c+1] = data[index][r][c];
        }
      }
      forward();
      backward(labels[index]);
    }

    // Evaluate on test set
    for(int test = 0; test < TESTS; test++) {
      for (int r = 0; r < INPUTWIDTH; r++) {
        for (int c = 0; c < INPUTWIDTH; c++) {
          // 1-padding offset
          input[r+1][c+1] = test_data[test][r][c];
        }
      }
      int label = test_labels[test];

      forward();

      countCases[label]++;
      if (label == maxScore)
        countRight[label]++;
    }

    // summary statistics:
    int testCases = 0;
    int successfulCases = 0;
    for (int i = 0; i < LABELS; i++) {
      printf("\n%d ratio correct: %0.2f", i, (double)countRight[i]/countCases[i]);
      testCases += countCases[i];
      successfulCases += countRight[i];
    }
    printf("\n%d tests, average correct: %0.2f\n", testCases,
        (double)successfulCases/testCases);
  }
}

int main()
{
  srand(time(0));
  init();
  if(read()) {
    printf("error loading files.\n");
    return 1;
  }
  //for(int i = 0; i < 500; i++)
  //  printf("%d, ", labels[i]);
  //printf("\n.\n");
  //for(int i = 0; i < 500; i++)
  //  printf("%d, ", test_labels[i]);
  //return 1;
  train();
  return 0;
}
