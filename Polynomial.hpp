#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H

#include <vector>
using namespace std;

template <class K>
class Polynomial {
    private:
        vector<K> coef;
        long dg;

        void deg_check() {
            dg = -1;

            for(long k=0; k < long(coef.size()); k++) {
                if(coef[k] != K(0)) dg = k;
            }
            coef.resize(dg + 1);
        }
    
    public:
        Polynomial <K> () {
            clear();
        }

        void clear() { coef.resize(1); dg=-1; coef[0] = K(0);}
        long deg() const {return dg;}

        K operator [] (long k) const {return get(k);}

        K get(long k) const {
            if (k < 0||k > dg) return K(0);
            return coef[k];
        }

        template <class J>
        Polynomial<K>(J a) {
            coef.clear();
            coef.resize(1);
            coef[0] = K(a);
            deg_check();
        }

        template <class J, class JJ, class JJJ>
        Polynomial <K>(J a, JJ b, JJJ c) {
            coef.clear();
            coef.resize(3);
            coef[2] = K(a);
            coef[1] = K(b);
            coef[0] = K(c);
            deg_check();
        }

};

// % End Of class


template <class K>
ostream& operator<<(ostream& os, const Polynomial<K>& P) {
    if (P.deg() <= 0) {
        os << " (" << P[0] << ")";
        return os;
    }

    for (long k=P.deg(); k >= 0; k--) {
        if(P[k] != K(0)) {
            if(k < P.deg()) os << " + ";
            os << "(" << P[k] << ")";
            if (k > 1) {
                os << "X^" << k;
                continue;
            }
            if (k==1) {
                os << "X";
            }
        }
    }
    return os;
}

#endif