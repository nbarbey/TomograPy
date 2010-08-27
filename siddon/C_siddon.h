#define SOLAR_RADIUS 695000. /* in km */
#define ARCSECOND_TO_RADIANS 0.00000484813681 /* = pi/(60*60*180) */
#define EPSILON 0.00001
#define ROIP_PIXEL_BASE 256
#define INF 1000000. /* en rayons solaires*/
#define DEGRE_TO_RADIAN 0.017453
#define N_MAX 1024/*max size of the cube in number of voxels*/


void init_C_siddon();
static PyObject *call_siddon(PyObject *self, PyObject *args);

int  not_doublematrix(PyArrayObject *mat);
double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin);
double **ptrvector(long n);
void free_Carrayptrs(double **v);


/* define the detector parameters */
typedef struct
{
  /* steps in radians*/
  float p1;
  float p2;
  /* size in pixels*/
  int n1;
  int n2;
  /* position of the sun center in pixels (can be a fraction of pixel */
  float s1;
  float s2;
}detector;

/* define the position of the detector */
typedef struct
{
  /* spherical coordinates*/
  float lon; /* longitude in radians */
  float lat; /* latitude in radians */
  float rol; /* roll angle in radians (angle between the image vertical and the Sun north pole taken positive clockwise )*/
  float d; /* distance in meters */
  /* cartesian coordinates in carrington */
  float xd;
  float yd;
  float zd;
}orbit;

/* define the parameters of the Region of interest of the object */
typedef struct
{
  /*size of the object cube in solar radii */
  float dx;
  float dy;
  float dz;
  /* steps in solar radius : 1 sr = 6.95E8 m */
  float px;
  float py;
  float pz;
  /* size in pixels */
  int nx;
  int ny;
  int nz;
  /* offset in solar radius : position of the firt pixel to center of rotation */
  /*bounds of the volume in solar radius*/
  float xmin;
  float xmax;
  float ymin;
  float ymax;
  float zmin;
  float zmax;  
}RoiO;

/*a basic cube structure to store object or projection data */
typedef struct
{
  int naxes[3];  /*size of each dimension*/
  float *** voxel; /*data*/
}cube;

int Siddon(
	   PyArrayObject*, /* 2D array containing the projection*/
	   int, /* current time index */
	   PyArrayObject*, /* 3D array containing the object */
	   orbit, /* orbit structure containing detector position parameters */
	   RoiO, /* volume structure containing object discretization parameters*/
	   detector, /* detector structure containing detector discretization parameters */
	   int /* projection or backprojection flag : 1 if backprojection */
	   );/* output 1 if no error */

int rotation_matrix(orbit, double [3][3]);
int define_unit_vector(double, double, double[3]);
int apply_rotation(double[3][3] , double[3], double[3]);
double distance_to_center(orbit, double *, double);
