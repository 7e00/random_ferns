#ifndef FERNS_H
#define FERNS_H

#include <stdlib.h>

#ifndef RAND_MAX
#define RAND_MAX 65535
#endif // RAND_MAX

class Binary_feature
{
protected:
    int fea_num;
    unsigned int *fea;
    Binary_feature() : fea_num(0), fea(0) {}
public:
    Binary_feature(int num) : fea_num(num)
    {
        const int nbits = sizeof(unsigned int)<<3;
        int len = (num + nbits-1)/nbits;
        fea = new unsigned int[len];
        memset(fea, 0, sizeof(unsigned int)*len);
    }
    Binary_feature(const Binary_feature &bf) : fea_num(bf.fea_num)
    {
        const int nbits = sizeof(unsigned int)<<3;
        int len = (fea_num + nbits-1)/nbits;
        fea = new unsigned int[len];
        memcpy(fea, bf.fea, sizeof(unsigned int)*len);
    }
    virtual ~Binary_feature()
    {
        if (fea)
            delete []fea;
    }
    unsigned int get_binary(int from, int to)
    {
        const int nbits = sizeof(unsigned int)<<3;
        int fp = from/(nbits);
        int foff = from%(nbits);
        int tp = to/(nbits);
        int toff = to%(nbits);
        if (fp == tp)
        {
            return (fea[fp]>>(nbits-1 - toff))&((1u<<(toff-foff+1))-1);
        }
        else
        {
            unsigned int res = fea[fp]&((1u<<(nbits-foff))-1);
            for (int i = fp+1; i < tp; ++i)
            {
                res <<= nbits;
                res += fea[i];
            }
            res <<= (toff+1);
            res += (fea[tp]>>(nbits-1-toff));
            return res;
        }
    }
    unsigned int get_binary()
    {
        return get_binary(0, fea_num-1);
    }
    void set_bit(int pos)
    {
        const int nbits = sizeof(unsigned int)<<3;
        int p = pos / nbits;
        int off = pos % nbits;
        fea[p] |= (1u<<(nbits-1-off));
    }
    void reset_bit(int pos)
    {
        const int nbits = sizeof(unsigned int)<<3;
        int p = pos / nbits;
        int off = pos % nbits;
        fea[p] &= ~(1u<<(nbits-1-off));
    }
    virtual Binary_feature *copy_self() const = 0;
    virtual void get_feature(double *vec, int veclen) = 0;
};

class Diff_Binary_feature : public Binary_feature
{
private:
    int *id;
    double *thrs;

public:
    Diff_Binary_feature(int num) : Binary_feature(num)
    {
        id = new int[fea_num<<1];
        thrs = new double[fea_num];
    }
    Diff_Binary_feature(int num, int veclen, double (*range)[2]) : Binary_feature(num)
    {
        set_random(veclen, range);
    }
    Diff_Binary_feature(int num, int veclen, double inf, double sup) : Binary_feature(num)
    {
        set_random(veclen, inf, sup);
    }
    Diff_Binary_feature(int num, int *aid, double *athrs) : Binary_feature(num)
    {
        id = new int[fea_num<<1];
        thrs = new double[fea_num];
        memcpy(id, aid, sizeof(int)*(fea_num<<1));
        memcpy(thrs, athrs, sizeof(double)*fea_num);
    }
    Diff_Binary_feature(const Diff_Binary_feature &dbf) : Binary_feature(dbf)
    {
        id = new int[fea_num<<1];
        thrs = new double[fea_num];
        memcpy(id, dbf.id, sizeof(int)*(fea_num<<1));
        memcpy(thrs, dbf.thrs, sizeof(double)*fea_num);
    }
    virtual ~Diff_Binary_feature()
    {
        delete []id;
        delete []thrs;
        id = 0;
        thrs = 0;
    }
    void set_param(int *aid, double *athrs)
    {
        memcpy(id, aid, sizeof(int)*(fea_num<<1));
        memcpy(thrs, athrs, sizeof(double)*fea_num);
    }
    void set_random(int veclen, double (*range)[2])
    {
        for (int i = 0; i < fea_num; ++i)
        {
            id[i<<1] = rand()%veclen;
            id[i<<1|1] = rand()%veclen;
            thrs[i] = range[i][0] + (range[i][1]-range[i][0])*(rand()%RAND_MAX) / (double)(RAND_MAX-1);
        }
    }
    void set_random(int veclen, double inf, double sup)
    {
        for (int i = 0; i < fea_num; ++i)
        {
            id[i<<1] = rand()%veclen;
            id[i<<1|1] = rand()%veclen;
            thrs[i] = inf + (sup-inf)*(rand()%RAND_MAX) / (double)(RAND_MAX-1);
        }
    }
    virtual Binary_feature *copy_self() const
    {
        //std::cout<<"in copyself of diff"<<std::endl;
        Diff_Binary_feature *dbf = new Diff_Binary_feature(*this);

        return dbf;
    }
    virtual void get_feature(double *vec, int veclen)
    {
        for (int i = 0; i < fea_num; ++i)
        {
            if ((vec[id[i<<1]] - vec[id[i<<1|1]]) < thrs[i])
                set_bit(i);
            else
                reset_bit(i);
        }
    }
};

class SingleFern
{
private:
    int depth; // fern depth(the number of binary feature the fern has)
    int class_num;

    // prob[i][j] = P(F=i|C=j)
    double *prob; //(1<<depth) * classnum

    // the class which the fern use to get the depth binary feature
    Binary_feature *bf;

    void preproc(int H)
    {
        if (bf)
            delete bf;
        bf = 0;
        int len = H*(1<<depth);
        if (prob)
        {
            if (H != class_num)
            {
                delete []prob;
                prob = new double[len];
            }
        }
        else
            prob = new double[len];
        class_num = H;
        for (int i = 0; i < len; prob[i++]=0);
    }

public:
    SingleFern(int fern_depth) : depth(fern_depth), class_num(0), prob(0), bf(0)
    {
    }
    ~SingleFern()
    {
        if (bf)
            delete bf;
        if (prob)
            delete []prob;
    }

    double train(double *X, int *C, int N, int K, int H, Binary_feature *getfea, int reg = 1)
    {
        preproc(H);
        int n = 1<<depth;
        bf = getfea->copy_self();
        double *vec = X;

        int *cnt = new int[H];
        memset(cnt, 0, sizeof(int)*H);
        for (int i = 0; i < N; ++i)
        {
            bf->get_feature(vec, K);
            unsigned int id = bf->get_binary(0, depth-1);
            ++prob[C[i]+id*H];
            ++cnt[C[i]];
            vec += K;
        }

        int idx = 0;
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < H; ++j)
            {
                prob[idx] = (prob[idx] + reg)/ ((double)cnt[j] + n*reg);
                ++idx;
            }
        }

        delete []cnt;

        return evaluate(X, C, N, K);
    }

    int classify(double *vec, int veclen, double *cprob = 0)
    {
        bf->get_feature(vec, veclen);
        unsigned int id = bf->get_binary(0, depth-1);
        int idx = id * class_num;
        double tprob = 0; //tprob = sigma(P(F=id|C=[0...class_num))
        double maxprob; //maxprob = max(P(F=id|C))
        int c;
        for (int i = 0; i < class_num; ++i)
        {
            if (i == 0 || prob[idx+i] > maxprob)
                maxprob = prob[idx+i], c = i;
            tprob += prob[idx+i];
        }
        if (cprob) //cprob = max(P(C|F=id))
            *cprob = maxprob / tprob;
        return c;
    }

    double evaluate(double *X, int *C, int N, int K, int *predict = 0)
    {
        double *vec = X;
        int rn = 0;
        for (int i = 0; i < N; ++i)
        {
            int pred = classify(vec, K);
            if (predict)
                predict[i] = pred;
            rn += (pred == C[i]);
            vec += K;
        }
        return (double)rn / (double)N;
    }
};

class RandomFerns
{
private:
    int m; // ferns' number
    int depth; //ferns' depth
    int class_num; //

    //prob[i][j][k] = log(P(F_k = i|C = j))
    double *prob; // m * (1<<depth) * class_num
    Binary_feature *bf;

    void preproc(int H)
    {
        if (bf)
            delete bf;
        bf = 0;
        int len = H*(1<<depth)*m;
        if (prob)
        {
            if (H != class_num)
            {
                delete []prob;
                prob = new double[len];
            }
        }
        else
            prob = new double[len];
        class_num = H;
        for (int i = 0; i < len; prob[i++]=0);
    }
public:
    RandomFerns(int fern_num, int fern_depth) : m(fern_num), depth(fern_depth), class_num(0), prob(0), bf(0)
    {
    }
    double train(double *X, int *C, int N, int K, int H, Binary_feature *getfea, int reg = 1)
    {
        preproc(H);
        int n = 1<<depth;
        bf = getfea->copy_self();
        double *vec = X;

        int *cnt = new int[H];
        memset(cnt, 0, sizeof(int)*H);
        for (int i = 0; i < N; ++i)
        {
            bf->get_feature(vec, K);
            for (int j = 0, k = 0; j < m; ++j,k+=depth)
            {
                unsigned id = bf->get_binary(k, k+depth-1);
                ++prob[j*n*H + id*H + C[i]];
            }
            ++cnt[C[i]];
            vec += K;
        }

        int idx = 0;
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                for (int k = 0; k < H; ++k)
                {
                    prob[idx] = (prob[idx] + reg)/ ((double)cnt[k] + n*reg);
                    prob[idx] = log(prob[idx]);
                    ++idx;
                }
            }
        }

        delete []cnt;

        return evaluate(X, C, N, K);
    }

    int classify(double *vec, int veclen, double *cprob = 0)
    {
        bf->get_feature(vec, veclen);
        int n = 1<<depth;
        double tprob = 0; //tprob = sigma(logp(F=feature|C=[0...class_num))
        double maxprob; //maxprob = max(logp(F=feature|C))
        int c;
        for (int i = 0; i < class_num; ++i)
        {
            double iprob = 0; //iprob = log(P(F=feature|C = i))
            for (int j = 0; j < m; ++j)
            {
                unsigned id = bf->get_binary(j*depth, j*depth+depth-1);
                iprob += prob[j*n*class_num + id*class_num + i];
            }
            if (i == 0 || iprob > maxprob)
                maxprob = iprob, c = i;

            tprob += iprob;
        }

        if (cprob)
            *cprob = maxprob/tprob;
        return c;
    }

    double evaluate(double *X, int *C, int N, int K, int *predict = 0)
    {
        double *vec = X;
        int rn = 0;
        for (int i = 0; i < N; ++i)
        {
            int pred = classify(vec, K);
            if (predict)
                predict[i] = pred;
            rn += (pred == C[i]);
            vec += K;
        }
        return (double)rn / (double)N;
    }
};

#endif // FERNS_H
