#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

class LinearRegression {
private:
    vector<vector<double>> X;
    vector<double> Y;
    vector<double> w;
    double b;

    double dot(const vector<double>& A, const vector<double>& B) const {
        double out = 0;
        for (size_t i = 0; i < A.size(); i++) {
            out += A[i] * B[i];
        }
        return out;
    }

    double computeCost() const {
        double cost_sum = 0;
        for (size_t i = 0; i < X.size(); i++) {
            double f_wb = dot(w, X[i]) + b;
            double cost = pow((f_wb - Y[i]), 2);
            cost_sum += cost;
        }
        return cost_sum / (2 * X.size());
    }

    vector<double> computeGradientW() const {
        vector<double> grad_w(w.size(), 0);
        for (size_t i = 0; i < X.size(); i++) {
            double f_wb = dot(w, X[i]) + b;
            for (size_t j = 0; j < w.size(); j++) {
                grad_w[j] += (f_wb - Y[i]) * X[i][j];
            }
        }
        for (size_t j = 0; j < grad_w.size(); j++) {
            grad_w[j] /= X.size();
        }
        return grad_w;
    }

    double computeGradientB() const {
        double grad_b = 0;
        for (size_t i = 0; i < X.size(); i++) {
            double f_wb = dot(w, X[i]) + b;
            grad_b += (f_wb - Y[i]);
        }
        return grad_b / X.size();
    }

public:
    LinearRegression(const vector<vector<double>>& x, const vector<double>& y)
        : X(x), Y(y), w(x[0].size(), 0), b(0) {}

    void fit(double alpha, int iterations) {
        for (int i = 0; i < iterations; i++) {
            vector<double> grad_w = computeGradientW();
            double grad_b = computeGradientB();
            for (size_t j = 0; j < w.size(); j++) {
                w[j] -= alpha * grad_w[j];
            }
            b -= alpha * grad_b;

            if (i % 100 == 0) {
                cout << "Iteration: " << i << " Cost: " << computeCost() << endl;
            }
        }
    }

    double predict(const vector<double>& x) const {
        return dot(w, x) + b;
    }

    void printParameters() const {
        cout << "Weights: ";
        for (double weight : w) {
            cout << weight << " ";
        }
        cout << endl;
        cout << "Bias: " << b << endl;
    }
};

int main() {
    vector<vector<double>> X = {{1, 2}, {2, 4}, {3, 6}, {4, 8}, {5, 10}};
    vector<double> Y = {2, 4, 6, 8, 10};

    LinearRegression lr(X, Y);
    lr.fit(0.01, 1000);
    lr.printParameters();

    vector<double> x = {6, 12};
    cout << "Prediction for the given data: " << lr.predict(x) << endl;

    return 0;
}
