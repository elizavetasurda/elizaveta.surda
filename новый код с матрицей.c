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
#define CANDIDATES 2000 //stolko tochek proveriem na every operations, грубо говоря поиск макс ei
#define MAX_DATA 200 //maksimalnoe sohr tochek

//parametrs GP
double length_scale[4] = {0.3, 0.5, 0.05, 0.05}; //masshtab po kazhdoi peremennoi
double noise = 1e-6; //chislennaia stabilizaciya

//struct information
typedef struct {
    double x[4];  //r l ts th
    double y;     //znachenie celevoi func (TEPER OBJECTIVE)
} Sample;

Sample dataset[MAX_DATA]; //massiv vseh experiment tochek
int data_count = 0; //tekyshee kolvo tochek

//random number of diapazon
double rand_uniform(double a, double b)
{
    return a + (b - a) * ((double)rand() / RAND_MAX); //generiruem random chislo v diapazone [a,b]
}

//---------------------------
//celevaia func (MASSA)
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

    return mass; //vozvrashaem massu bez penalty

    /*
    // Старый код mass_function (объединялся с penalty)
    double Vmat = (4.0 / 3.0) * PI * (pow(R + Th, 3) - pow(R, 3)) + PI * L * (pow(R + Ts, 2) - pow(R, 2));
    double mass = rho * Vmat;
    return mass + 0; // раньше штраф был внутри
    */
}

//---------------------------
//shtraf otdelno (teper myagkii)
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

    //mягкий штраф: вычисляем na skolko nepravilno
    if (V < Vmin)
        penalty += 1e7 * (Vmin - V); //bol'shiy penalty esli obem menshe minimuma

    if (sigma > sigma_allow)
        penalty += 1e7 * (sigma - sigma_allow) / sigma_allow; //bol'shiy penalty esli napryazhenie prevyshaet dopustimoe

    return penalty;

    /*
    // Старый код penalty_function (не мягкий, сразу добавлял фиксированное значение)
    double penalty = 0.0;
    if (V < Vmin)
        penalty += 1e7;
    if (sigma > sigma_allow)
        penalty += 1e7;
    return penalty;
    */
}

//---------------------------
//polnaya celevaia func (mass + shtraf)
double objective(double x[4])
{
    //sum mass + penalty
    return mass_function(x) + penalty_function(x);

    /*
    // Старый код objective (всё считалось сразу)
    double R = x[0], L = x[1], Ts = x[2], Th = x[3];
    double V = (4.0 / 3.0) * PI * pow(R,3) + PI*R*R*L;
    double Vmat = (4.0 / 3.0) * PI * (pow(R+Th,3) - pow(R,3)) + PI*L*(pow(R+Ts,2)-pow(R,2));
    double mass = rho*Vmat;
    double sigma1 = P*R/(2.0*Th), sigma2 = P*R/Ts;
    double sigma = sigma1 > sigma2 ? sigma1 : sigma2;
    double penalty = 0.0;
    if(V<Vmin) penalty+=1e7;
    if(sigma>sigma_allow) penalty+=1e7;
    return mass+penalty;
    */
}

//---------------------------
//gaussian process 
// k(x1,x2) = exp( -0.5 * ||x1-x2||^2 / l^2 )
double kernel(double a[4], double b[4])
{
    double sum = 0;
    for (int i = 0; i < 4; i++)
    {
        double d = (a[i] - b[i]) / length_scale[i]; //s uchetom masshtaba
        sum += d * d;
    }
    return exp(-0.5 * sum); //vozvrashchaem znachenie kernel

    /*
    // Старый kernel
    double sum=0;
    for(int i=0;i<4;i++){double d=a[i]-b[i]; sum+=d*d;}
    return exp(-sum);
    */
}

//---------------------------
//Cholesky i solve_system
int cholesky(double K[MAX_DATA][MAX_DATA], double L[MAX_DATA][MAX_DATA], int n)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            L[i][j] = 0;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            double sum = 0;
            for (int k = 0; k < j; k++)
                sum += L[i][k] * L[j][k];

            if (i == j)
            {
                double val = K[i][i] - sum;
                if (val <= 0) return 0;
                L[i][j] = sqrt(val); //diagonal element
            }
            else
            {
                L[i][j] = (K[i][j] - sum) / L[j][j]; //off-diagonal element
            }
        }
    }
    return 1;
}

// solve L L^T x = b
void solve_system(double L[MAX_DATA][MAX_DATA], double b[MAX_DATA], double x[MAX_DATA], int n)
{
    double y[MAX_DATA];

    //forward substitution
    for (int i = 0; i < n; i++)
    {
        double sum = 0;
        for (int k = 0; k < i; k++) sum += L[i][k] * y[k];
        y[i] = (b[i] - sum) / L[i][i];
    }

    //backward substitution
    for (int i = n - 1; i >= 0; i--)
    {
        double sum = 0;
        for (int k = i + 1; k < n; k++) sum += L[k][i] * x[k];
        x[i] = (y[i] - sum) / L[i][i];
    }
}

//---------------------------
//predskazanie vozvr: 1.srednie znach 2.neopredelennost
void gp_predict(double x[4], double *mu, double *sigma)
{
    double K[MAX_DATA][MAX_DATA];
    double L[MAX_DATA][MAX_DATA];
    double y_vec[MAX_DATA];
    double alpha[MAX_DATA];
    double k_star[MAX_DATA];

    //normalizaciya y
    double mean_y = 0;
    for(int i=0;i<data_count;i++) mean_y+=dataset[i].y;
    mean_y/=data_count;

    double std_y=0;
    for(int i=0;i<data_count;i++) std_y+=pow(dataset[i].y-mean_y,2);
    std_y=sqrt(std_y/data_count)+1e-12;

    for(int i=0;i<data_count;i++) y_vec[i]=(dataset[i].y-mean_y)/std_y;

    //kovariacionnaya matrica
    for(int i=0;i<data_count;i++)
        for(int j=0;j<data_count;j++)
        {
            K[i][j]=kernel(dataset[i].x,dataset[j].x);
            if(i==j) K[i][j]+=noise;
        }

    cholesky(K,L,data_count);
    solve_system(L,y_vec,alpha,data_count);

    //korrelyaciya novoi tochki s obuchayushimi
    for(int i=0;i<data_count;i++)
        k_star[i]=kernel(x,dataset[i].x);

    double mean=0;
    for(int i=0;i<data_count;i++)
        mean+=k_star[i]*alpha[i];

    double v[MAX_DATA];
    for(int i=0;i<data_count;i++)
    {
        double sum=0;
        for(int k=0;k<i;k++) sum+=L[i][k]*v[k];
        v[i]=(k_star[i]-sum)/L[i][i];
    }

    double var=kernel(x,x);
    for(int i=0;i<data_count;i++) var-=v[i]*v[i];
    if(var<1e-12) var=1e-12;

    *mu=mean*std_y+mean_y; //vozvrashchaem srednee predskazanie
    *sigma=sqrt(var)*std_y; //vozvrashchaem neopredelennost

    /*
    // Старый gp_predict (упрощённый)
    double k_star[MAX_DATA]; double k_sum=0;
    for(int i=0;i<data_count;i++){ k_star[i]=kernel(x,dataset[i].x); k_sum+=k_star[i]; }
    double mean=0; for(int i=0;i<data_count;i++) mean+=k_star[i]*dataset[i].y;
    mean/=(k_sum+1e-8);
    double var=1.0; for(int i=0;i<data_count;i++) var-=k_star[i]*k_star[i];
    if(var<1e-6)var=1e-6;
    *mu=mean; *sigma=sqrt(var);
    */
}

//---------------------------
// expected improvement
double normal_pdf(double x){ return exp(-0.5*x*x)/sqrt(2*PI); }
double normal_cdf(double x){ return 0.5*(1+erf(x/sqrt(2))); }

double expected_improvement(double mu,double sigma,double best)
{
    if(sigma<1e-12) return 0;
    double Z=(best-mu)/sigma; //nasколько tochka luchshe tekushchego minimuma
    double EI=(best-mu)*normal_cdf(Z)+sigma*normal_pdf(Z); //polza ot uluchsheniya + polza ot issledovaniya
    if(EI<0) EI=0;
    return EI;
}

//---------------------------
//generate tochky
void random_point(double x[4])
{
    for(int i=0;i<4;i++)
        x[i]=rand_uniform(bounds[i][0],bounds[i][1]); //generiruem random point v granitsah
}

//---------------------------
//main
int main()
{
    srand(time(NULL));
    printf("Bayesian Optimization - Pressure Vessel\n\n");

    //start tochki
    for(int i=0;i<INIT_POINTS;i++)
    {
        random_point(dataset[i].x);
        dataset[i].y=objective(dataset[i].x); //schitaem func
    }

    data_count=INIT_POINTS;

    //osnovnoi cikl
    for(int iter=0;iter<ITERATIONS;iter++)
    {
        if(data_count>=MAX_DATA) break;

        double best=dataset[0].y;
        for(int i=1;i<data_count;i++) if(dataset[i].y<best) best=dataset[i].y;

        double best_ei=-1;
        double best_candidate[4];

        for(int c=0;c<CANDIDATES;c++)
        {
            double x[4];
            random_point(x);

            double mu,sigma;
            gp_predict(x,&mu,&sigma);

            double ei=expected_improvement(mu,sigma,best); //vykhod EI
            if(ei>best_ei)
            {
                best_ei=ei;
                for(int k=0;k<4;k++) best_candidate[k]=x[k]; //zapominayem luchshuyu kandidat
            }
        }

        dataset[data_count].y=objective(best_candidate); //vychislyaem realnuyu funct
        for(int k=0;k<4;k++) dataset[data_count].x[k]=best_candidate[k];
        data_count++;

        printf("Iteration %d | Best objective = %.2f\n", iter+1,best);
    }

    int best_id=0;
    for(int i=1;i<data_count;i++)
        if(dataset[i].y<dataset[best_id].y) best_id=i;

    printf("\nFINAL RESULT:\n");
    printf("R  = %.4f m\n", dataset[best_id].x[0]);
    printf("L  = %.4f m\n", dataset[best_id].x[1]);
    printf("Ts = %.4f m\n", dataset[best_id].x[2]);
    printf("Th = %.4f m\n", dataset[best_id].x[3]);
    printf("Mass (pure) = %.2f kg\n", mass_function(dataset[best_id].x)); //mass bez penalty
    printf("Penalty     = %.2f\n", penalty_function(dataset[best_id].x));
    printf("Objective   = %.2f\n", dataset[best_id].y);

    return 0;
}
