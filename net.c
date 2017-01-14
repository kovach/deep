#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/*
 *   currently this is a softmax classifier for the MNIST dataset:
 *       http://cis.jhu.edu/~sachin/digit/digit.html
 *
 *   typical output:
 *     ===== EPOCH: 0 =====
 *     ...
 *     ===== EPOCH: 14 =====
 *     EOF at 10000!
 *     0 ratio correct: 0.97
 *     1 ratio correct: 0.97
 *     2 ratio correct: 0.90
 *     3 ratio correct: 0.90
 *     4 ratio correct: 0.93
 *     5 ratio correct: 0.88
 *     6 ratio correct: 0.95
 *     7 ratio correct: 0.93
 *     8 ratio correct: 0.88
 *     9 ratio correct: 0.92
 *
 *   TODO:
 *     - add convolution layer
 *     - code generation
 */

#define L1DEPTH 1
#define WINDOW 3
#define LABELS 10
#define INPUTWIDTH 28
#define DROPOUTWIDTH 28
#define PADDING 0

// parameters/layer values from forward pass
double f0[L1DEPTH][WINDOW*WINDOW];
double full[LABELS][L1DEPTH][DROPOUTWIDTH][DROPOUTWIDTH];
double bias[LABELS];
double input[1][INPUTWIDTH+PADDING][INPUTWIDTH+PADDING];
double L1[L1DEPTH][INPUTWIDTH][INPUTWIDTH];
double dropout[L1DEPTH][DROPOUTWIDTH][DROPOUTWIDTH];
double softmax[LABELS];

// parameter/layer gradients
double _f0[L1DEPTH][WINDOW*WINDOW];
double _full[LABELS][L1DEPTH][DROPOUTWIDTH][DROPOUTWIDTH];
double _bias[LABELS];
double _input[1][INPUTWIDTH+PADDING][INPUTWIDTH+PADDING];
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

/* Init weights, Forward Pass, Backward pass */
void init()
{
  for (int out = 0; out < LABELS; out++) {
    bias[out] = 0.0;
  for (int d = 0; d < L1DEPTH; d++) {
  for (int r = 0; r < DROPOUTWIDTH; r++) {
  for (int c = 0; c < DROPOUTWIDTH; c++) {
    full[out][d][r][c] = ((double)rand())/((double)RAND_MAX)*2-1;
  }}}}
}
void forward()
{
  // TODO layers before dropout

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
  // compute gradient at fully connected layer
  // softmax = full * dropout + bias
  for (int out = 0; out < LABELS; out++) {
    _bias[out] = _softmax[out];
    // L2 regularize
    bias[out] *= 0.999999;
    bias[out] += _bias[out] / 3.3;
    for (int d = 0; d < L1DEPTH; d++) {
    for (int r = 0; r < DROPOUTWIDTH; r++) {
    for (int c = 0; c < DROPOUTWIDTH; c++) {
      _full[out][d][r][c] = _softmax[out] * dropout[d][r][c];
      // L2 regularize
      full[out][d][r][c] *= 0.999999;
      full[out][d][r][c] += _full[out][d][r][c] / (28.0*3.3);
    }}}
  }
  // TODO layers before dropout
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
    int maxExamples = 15000;
    for(int index = epoch > 0 ? 0 : 0; index < maxExamples; index++) {
      label = (index % LABELS);

      // load an image
      for(int i = 0; i < INPUTWIDTH*INPUTWIDTH; i++) {
        int b = fgetc(fp[label]);
        if (b == EOF) {
          printf("EOF at %d!\n", index);
          goto done;
        }
        ((double *)dropout)[i] = ((double)(b-128))/255.0;
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
