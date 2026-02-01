#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define PI 3.14159265359

//parametrs zadachi

double P = 4e6;           //давление
double Vmin = 0.4;        //минимальный объем
double rho = 7800;        //плотность
double sigma_allow = 9.5e6;  //допустимое напряжение

//granits proektnih peremennih

double bounds[4][2] = {
    {0.2, 1.0},     //r
    {0.2, 1.8},     //l
    {0.01, 0.15},   //ts толщина цилиндра
    {0.01, 0.1}     //th толщина полусфер
};

//parametrs algoritm

#define INIT_POINTS 15 //kolvo start random experiments
#define ITERATIONS 40 //itterations of bayesian optimization
#define CANDIDATES 2000 //stolko tochek proveriem na every operations,грубо говоря поиск макс ei
#define MAX_DATA 200 //maksimalnoe sohr tochek

//struct information

typedef struct {
    double x[4];  //r l ts th
    double y;     //znachenie celevoi func(mass)
} Sample;

Sample dataset[MAX_DATA]; //massiv vseh experiment tochek
int data_count = 0; //tekyshee kolvo tochek

//random number of diapazon

double rand_uniform(double a, double b)
{
    return a + (b - a) * ((double)rand() / RAND_MAX);
}

//celevaia func (TOLKO MASSA)

double mass_function(double x[4])
{
    double R = x[0];
    double L = x[1];
    double Ts = x[2];
    double Th = x[3];
    //volume matirial
    double Vmat =
        (4.0 / 3.0) * PI * (pow(R + Th, 3) - pow(R, 3)) + 
        PI * L * (pow(R + Ts, 2) - pow(R, 2));
    //massa
    double mass = rho * Vmat;

    return mass;
}

//shtraf otdelno

double penalty_function(double x[4])
{
    double R = x[0];
    double L = x[1];
    double Ts = x[2];
    double Th = x[3];

    double penalty = 0.0;

    //volume sosyda
    double V = (4.0 / 3.0) * PI * pow(R, 3) + PI * R * R * L;

    //voltage
    double sigma1 = P * R / (2.0 * Th);
    double sigma2 = P * R / Ts;
    double sigma = sigma1 > sigma2 ? sigma1 : sigma2;

    if (V < Vmin)
        penalty += 1e7;

    if (sigma > sigma_allow)
        penalty += 1e7;

    return penalty;
}

//polnaya celevaia func (mass + shtraf)

double objective(double x[4])
{
    return mass_function(x) + penalty_function(x);
}

//gaussian process 
// k(x1,x2) = exp( -||x1-x2||^2 )

double kernel(double a[4], double b[4]) //func shodstva 2h tochek
{
    double sum = 0;
    for (int i = 0; i < 4; i++)
    {
        double d = a[i] - b[i]; //- kooordinat
        sum += d * d;//evklidovo rasstoyanie
    }
    return exp(-sum);
}

//predskazanie vozvr: 1.srednie znach 2.neopredelennost

void gp_predict(double x[4], double *mu, double *sigma)
{
    double k_star[MAX_DATA];
    double k_sum = 0;
    //корреляция новой точки с обучающими
    for (int i = 0; i < data_count; i++)
    {
        k_star[i] = kernel(x, dataset[i].x);
        k_sum += k_star[i];
    }
    //sr znach
    double mean = 0;
    for (int i = 0; i < data_count; i++)
    {
        mean += k_star[i] * dataset[i].y;
    }
    mean /= (k_sum + 1e-8);
    //neopredellenost
    double var = 1.0;
    for (int i = 0; i < data_count; i++)
    {
        var -= k_star[i] * k_star[i]; //чем больше похожих точек тем меньше неопределенность 
    }
    if (var < 1e-6)
        var = 1e-6;

    *mu = mean;
    *sigma = sqrt(var); //отклонение 
}

// expected improvement 

double normal_pdf(double x)
{
    return exp(-0.5 * x * x) / sqrt(2 * PI);
}
double normal_cdf(double x)
{
    return 0.5 * (1 + erf(x / sqrt(2)));
}
double expected_improvement(double mu, double sigma, double best)
{
    if (sigma < 1e-6)
        return 0;

    double Z = (best - mu) / sigma; //насколько новая точка лучше текущего минимума 

    double EI = (best - mu) * normal_cdf(Z) + sigma * normal_pdf(Z); //польза улучшения + польза исследования
    return EI; //итоговая польза
}

//generate tochky

void random_point(double x[4])
{
    for (int i = 0; i < 4; i++)
    {
        x[i] = rand_uniform(bounds[i][0], bounds[i][1]); 
    }
}

int main()
{
    srand(time(NULL)); //zapysk generatora rand chisel
    printf("Bayesian Optimization - Pressure Vessel\n\n");
    
    //start tochki

    for (int i = 0; i < INIT_POINTS; i++)
    {
        random_point(dataset[i].x);

        //GP obuchaetsya TOLKO na masse

        dataset[i].y = mass_function(dataset[i].x);
    }

    data_count = INIT_POINTS;

    //osnovnoi cikl

    for (int iter = 0; iter < ITERATIONS; iter++) //iteracii Bayesian Optimization
    {
        double best = dataset[0].y; //tekyshii min massy
        for (int i = 1; i < data_count; i++) 
            if (dataset[i].y < best)
                best = dataset[i].y;

        double best_ei = -1;
        double best_candidate[4];

        //generate candidats
        for (int c = 0; c < CANDIDATES; c++)
        {
            double x[4];
            random_point(x);

            double mu, sigma;
            gp_predict(x, &mu, &sigma); //predskazanie
            double ei = expected_improvement(mu, sigma, best); //насколько выгодна точка
            if (ei > best_ei) //ищем максимум
            {
                best_ei = ei;

                for (int k = 0; k < 4; k++)
                    best_candidate[k] = x[k];
            }
        }

        //вычисляем реальную функцию (TOLKO MASSA dlya GP)

        dataset[data_count].y = mass_function(best_candidate);

        for (int k = 0; k < 4; k++)
            dataset[data_count].x[k] = best_candidate[k];

        data_count++;

        printf("Iteration %d | Best mass = %.2f kg\n", iter + 1, best);
    }

    int best_id = 0;

    for (int i = 1; i < data_count; i++)
        if (dataset[i].y < dataset[best_id].y)
            best_id = i;

    printf("\nFINAL RESULT:\n");
    printf("R  = %.4f m\n", dataset[best_id].x[0]);
    printf("L  = %.4f m\n", dataset[best_id].x[1]);
    printf("Ts = %.4f m\n", dataset[best_id].x[2]);
    printf("Th = %.4f m\n", dataset[best_id].x[3]);

    printf("Mass (pure) = %.2f kg\n", dataset[best_id].y);
    printf("Penalty     = %.2f\n", penalty_function(dataset[best_id].x));
    printf("Objective   = %.2f\n", objective(dataset[best_id].x));

    return 0;
}
