#define EPSILON 0.00001
#define INF 1000000. /* en rayons solaires*/

/* define the detector parameters */
typedef struct
{
  /* steps in radians*/
  double p1;
  double p2;
  /* size in pixels*/
  int n1;
  int n2;
  /* position of the sun center in pixels (can be a fraction of pixel */
  double s1;
  double s2;
}detector;

/* define the position of the detector */
typedef struct
{
  /* spherical coordinates*/
  double lon; /* longitude in radians */
  double lat; /* latitude in radians */
  double rol; /* roll angle in radians (angle between the image vertical and the Sun north pole taken positive clockwise )*/
  double d; /* distance in meters */
  /* cartesian coordinates in carrington */
  double M[3];
}orbit;

/* define the parameters of the Region of interest of the object */
typedef struct
{
  /*size of the object cube in solar radii */
  double d[3];
  /* steps in solar radius : 1 sr = 6.95E8 m */
  double p[3];
  /* size in pixels */
  int n[3];
  /* offset in solar radius : position of the firt pixel to center of rotation */
  /*bounds of the volume in solar radius*/
  double min[3];
  double max[3];
}RoiO;

void init_C_siddon();
static PyObject *call_siddon_sun(PyObject *self, PyObject *args);
static PyObject *call_siddon(PyObject *self, PyObject *args);

int  not_doublematrix(PyArrayObject *mat);
double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin);
double **ptrvector(long n);
void free_Carrayptrs(double **v);

double min3(double,double,double);
double max3(double,double,double);
int signe(double);
void Compare(double*,double*,double,double);

int rotation_matrix(orbit, double [3][3]);
int define_unit_vector(double, double, double[3]);
int apply_rotation(double[3][3] , double[3], double[3]);
double distance_to_center(orbit, double *, double);

int SiddonSun(
	   PyArrayObject*, /* 2D array containing the projection*/
	   int, /* current time index */
	   PyArrayObject*, /* 3D array containing the object */
	   orbit, /* orbit structure containing detector position parameters */
	   RoiO, /* volume structure containing object discretization parameters*/
	   detector, /* detector structure containing detector discretization parameters */
	   int /* projection or backprojection flag : 1 if backprojection */
	   );/* output 1 if no error */

int Siddon(
	   PyArrayObject*, /* 2D array containing the projection*/
	   int, /* current time index */
	   PyArrayObject*, /* 3D array containing the object */
	   orbit, /* orbit structure containing detector position parameters */
	   RoiO, /* volume structure containing object discretization parameters*/
	   detector, /* detector structure containing detector discretization parameters */
	   int /* projection or backprojection flag : 1 if backprojection */
	   );/* output 1 if no error */
