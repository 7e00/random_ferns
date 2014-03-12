//#include <armadillo>

#ifndef RAND_MAX
#define RAND_MAX 65535

class Binary_feature
{
protected:
    int (*get_binary_func)(double *vec, int veclen);
    Binary_feature() : get_binary_func(0) {}
public:
    Binary_feature(int (*func)(double *vec, int veclen))
    {
        get_binary_func = func;
    }
    virtual ~Binary_feature()
    {
    }
    virtual int get_binary(double *vec, int veclen)
    {
        return get_binary_func(vec, veclen);
    }
};

class Diff_Binary_feature : public Binary_feature
{
private:
    int s;
    int *id;
    double *thrs;
    
public:
    Diff_Binary_feature(int depth)
    {
        s = depth;
        id = new int[s<<1];
        thrs = new double[s];
    }
    Diff_Binary_feature(int depth, int *aid, double *athrs)
    {
        s = depth;
        memcpy(id, aid, sizeof(int)*(s<<1));
        memcpy(thrs, athrs, sizeof(double)*s);
    }
    Diff_Binary_feature(const Diff_Binary_feature &dbf)
    {
        s = dbf.s;
        id = new int[s<<1];
        thrs = new double[s];
        memcpy(id, dbf.id, sizeof(int)*(s<<1));
        memcpy(thrs, dbf.thrs, sizeof(double)*s);
    }
    virtual ~Diff_Binary_feature()
    {
        delete []id;
        delete []thrs;
        id = 0;
        thrs = 0;
        s = 0;
    }
    void set_param(int *aid, double *athrs)
    {
        memcpy(id, aid, sizeof(int)*(s<<1));
        memcpy(thrs, athrs, sizeof(double)*s);
    }
    void set_random(int veclen, double (*range)[2])
    {
        for (int i = 0; i < s; ++i)
        {
            id[i<<1] = rand()%veclen;
            id[i<<1|1] = rand()%veclen;
            thrs[i] = range[i][0] + (range[i][1]-range[i][0])*(rand()%RANDMAX) / (double)(RAND_MAX-1);
        }
    }
    void set_random(int veclen, double inf, double sup)
    {
        for (int i = 0; i < s; ++i)
        {
            id[i<<1] = rand()%veclen;
            id[i<<1|1] = rand()%veclen;
            thrs[i] = inf + (sup-inf)*(rand()%RANDMAX) / (double)(RAND_MAX-1);
        }
    }
    virtual int get_binary(double *vec, int veclen)
    {
        num = 0;
        for (int i = 0; i < s; ++i)
        {
            num <<= 1;
            num += ((vec[id[i<<1]] - vec[id[i<<1|1]]) < thrs[i]);
        }
        return num;
    }
    virtual int operator()(double *vec, int veclen)
    {
        return get_binary(vec, veclen);
    }
};
class SingleFern
{
public:
    typedef int (*get_binary_feature)(double *, int);
private:
    int depth;
    int class_num;
    double *prob;
    Binary_feature *bf;

    void preproc(int H)
    {
        if (bf)
            delete bf;
        bf = 0;
        if (prob)
            delete []prob;
        class_num = H;
        int len = H*(1<<depth);
        prob = new double[len];
        for (int i = 0; i < len; prob[i++]=0);
    }

public:
    SingleFern(int depth)
    {
        this->depth = depth;
        prob = 0;
        bf = 0;
    }
    ~SingleFern()
    {
        if (bf)
            delete bf;
        if (prob)
            delete []prob;
    }
    
    template<class Get_Binary_Feature> void train(double *X, int *C, int N, int K, int H, Get_Binary_Feature &func, int reg = 1)
    {
        preproc(H);
        int n = 1<<depth;
        bf = new Get_Binary_Feature(func);
        double *vec = X;
        
        int *cnt = new int[H];
        for (int i = 0; i < N; ++i)
        {
            int id = bf.get_binary(vec, K);
            ++prob[C[i]+id*H];
            ++cnt[C[i]];
        }
        
        int idx = 0;
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < H; ++j)
            {
                prob[idx] = (prob[idx] + reg)/ ((double)cnt[j] + n*reg);
                prob[idx] = log(prob[idx]);
                ++idx;
        }
    }
    void train(double *X, int *C, int N, int K, int H, get_binary_feature func)
    {
        train(X, C, N, K, H, Binary_feature(func));
    }

    int classify(double *vec, int K)
    {
        int id = bf.get_binary(vec, K);
        int idx = id * class_num;
        double maxprob;
        int c;
        for (int i = 0; i < class_num; ++i)
        {
            if (i == 0 || maxprob < prob[idx])
                maxprob = prob[idx], c = i;
            ++idx;
        }
        return c;
    }
};
