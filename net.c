#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/*
 *   this is a classifier for the MNIST dataset:
 *       http://cis.jhu.edu/~sachin/digit/digit.html
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
 *     0) softmax:
 *       ===== EPOCH: 14 =====
 *       EOF at 10000!
 *       0 ratio correct: 0.97
 *       1 ratio correct: 0.97
 *       2 ratio correct: 0.90
 *       3 ratio correct: 0.90
 *       4 ratio correct: 0.93
 *       5 ratio correct: 0.88
 *       6 ratio correct: 0.95
 *       7 ratio correct: 0.93
 *       8 ratio correct: 0.88
 *       9 ratio correct: 0.92
 *       ~30 second runtime
 *
 *     1) with 1 linear convolution layer (6 filters):
 *       ===== EPOCH: 14 =====
 *       0 ratio correct: 0.96
 *       1 ratio correct: 0.94
 *       2 ratio correct: 0.88
 *       3 ratio correct: 0.87
 *       4 ratio correct: 0.90
 *       5 ratio correct: 0.85
 *       6 ratio correct: 0.93
 *       7 ratio correct: 0.90
 *       8 ratio correct: 0.85
 *       9 ratio correct: 0.85
 *       ./net  284.58s user 0.09s system 99% cpu 4:44.77 total
 *
 *
 *     2) with 1 convolution layer and relu nonlinearity (6 filters):
 *     ...
 *     ===== EPOCH: 6 =====
 *     EOF at 10000!
 *     0 ratio correct: 0.97
 *     1 ratio correct: 0.96
 *     2 ratio correct: 0.93
 *     3 ratio correct: 0.93
 *     4 ratio correct: 0.94
 *     5 ratio correct: 0.93
 *     6 ratio correct: 0.96
 *     7 ratio correct: 0.94
 *     8 ratio correct: 0.92
 *     9 ratio correct: 0.92
 *     ...
 *     ===== EPOCH: 14 =====
 *     EOF at 10000!
 *     0 ratio correct: 0.99
 *     1 ratio correct: 0.98
 *     2 ratio correct: 0.97
 *     3 ratio correct: 0.98
 *     4 ratio correct: 0.98
 *     5 ratio correct: 0.99
 *     6 ratio correct: 0.99
 *     7 ratio correct: 0.98
 *     8 ratio correct: 0.98
 *     9 ratio correct: 0.97
 *     ./net  281.79s user 0.17s system 99% cpu 4:42.24 total
 *
 *
 *
 *   TODO:
 *     - display filters learned
 *     - second convolution layer?
 *     - code generation
 */

#define L1DEPTH 6
#define WINDOW 3
#define LABELS 10
#define INPUTWIDTH 28
#define DROPOUTWIDTH 28

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
int label;
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
  printf("\n");
  //for (int out = 0; out < LABELS; out++)
  //  printf("%0.1f, ", _softmax[out]);
  //printf("\n");
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

void printLabelling()
{
  printf("input label: %d, output label: %d, score: %0.2f\n", label, maxScore, softmaxDistribution[maxScore]);
}

double randun()
{
  return ((double)rand())/((double)RAND_MAX)*2.0-1.0;
}

/* Init weights, Forward Pass, Backward pass */
void init()
{
  // bias, weights
  for (int out = 0; out < LABELS; out++) {
    bias[out] = 0.0;
  for (int d = 0; d < L1DEPTH; d++) {
  for (int r = 0; r < DROPOUTWIDTH; r++) {
  for (int c = 0; c < DROPOUTWIDTH; c++) {
    full[out][d][r][c] = randun();
  }}}}
  // convolution filters
  for (int d = 0; d < L1DEPTH; d++) {
    for(int wr = 0; wr < WINDOW; wr++) {
    for(int wc = 0; wc < WINDOW; wc++) {
      // TODO add positive bias?
      f1[d][wr][wc] = randun();
    }}
  }
  // input padding
  for (int i = 0; i < INPUTWIDTH+1; i++) {
    input[0][i] = 0.0;
    input[i][INPUTWIDTH-1] = 0.0;
    input[INPUTWIDTH-1][i+1];
    input[i+1][0] = 0.0;
  }
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

void backward()
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
    // L2 regularize
    bias[out] *= 0.999999;
    bias[out] += _bias[out] / 3.3;
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
  for (int d = 0; d < L1DEPTH; d++) {
  for (int r = 0; r < DROPOUTWIDTH; r++) {
  for (int c = 0; c < DROPOUTWIDTH; c++) {
  for (int out = 0; out < LABELS; out++) {
      full[out][d][r][c] *= 0.999999;
      full[out][d][r][c] += _full[out][d][r][c] / (28.0*3.3);
  }}}}
  for (int d = 0; d < L1DEPTH; d++) {
    for(int wr = 0; wr < WINDOW; wr++) {
      for(int wc = 0; wc < WINDOW; wc++) {
        f1[d][wr][wc] *= 0.999999;
        f1[d][wr][wc] += _f1[d][wr][wc]/100;
      }}}
}

void read()
{
  FILE *fp[LABELS];

  int openFiles;

  for (int epoch = 0; epoch < 15; epoch++) {
    // (re)open files
    printf("===== EPOCH: %d =====\n", epoch);
    for(openFiles = 0; openFiles < LABELS; openFiles++) {
      int digit = openFiles;
      char filename[5];
      sprintf(filename, "data%d", digit);
      fp[digit] = fopen(filename, "r");
      if (fp == NULL) {
        printf("error: couldn't open data%d\n", digit);
        goto done;
      }
    }

    // record improvements each epoch
    for (int i = 0; i < LABELS; i++) {
      countRight[i] = 0;
      countCases[i] = 0;
    }

    // Train, alternating between digit types
    int maxExamples = 50000;
    for(int index = 0; index < maxExamples; index++) {
      label = (index % LABELS);

      // load an image
      for (int r = 0; r < INPUTWIDTH; r++) {
        for (int c = 0; c < INPUTWIDTH; c++) {
          int b = fgetc(fp[label]);
          if (b == EOF) {
            printf("EOF at %d!\n", index);
            goto done;
          }
          // 1-padding offset
          input[r+1][c+1] = ((double)(b-128))/255.0;
        }
      }

      // TODO normalize image

      forward();
      backward();

      countCases[label]++;
      if (label == maxScore)
        countRight[label]++;
    }
done:
    for (int i = 0; i < LABELS; i++) {
      printf("%d ratio correct: %0.2f\n", i, (double)countRight[i]/countCases[i]);
    }

    //printThetaNorm();

    // Close files
    for(int digit = 0; digit < openFiles; digit++)
      fclose(fp[digit]);
  }
}

int main()
{
  srand(time(0));
  init();
  read();
  return 0;
}
