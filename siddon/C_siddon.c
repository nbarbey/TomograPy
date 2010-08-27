/*
  Siddon algorithm  and python warper It performs  3D conic projection
  and backprojection for tomography applications
*/

#include "Python.h"
#include "arrayobject.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "C_siddon.h"
#include <omp.h>

#define DIND1(a, i) *((double *) PyArray_GETPTR1(a, i))
#define DIND2(a, i, j) *((double *) PyArray_GETPTR2(a, i, j))
#define DIND3(a, i, j, k) *((double *) PyArray_GETPTR3(a, i, j, k))
#define FIND1(a, i) (float)*((float *) PyArray_GETPTR1(a, i))
#define FIND2(a, i, j) *((float *) PyArray_GETPTR2(a, i, j))
#define FIND3(a, i, j, k) *((float *) PyArray_GETPTR3(a, i, j, k))

static PyMethodDef _C_siddonMethods[] = {
  {"siddon", call_siddon, METH_VARARGS},
  {NULL, NULL}     /* Sentinel - marks the end of this structure */
};

void init_C_siddon()  {
  (void) Py_InitModule("_C_siddon", _C_siddonMethods);
  import_array();  // Must be present for NumPy.  Called first after above line.
  Py_Initialize();
}

static PyObject *call_siddon(PyObject *self, PyObject *args)
{
  /* Input and output matrices to be extracted from args */
  PyArrayObject *data, *map;
  PyObject *header, *cube_header;
  orbit orbit;
  detector detector;
  RoiO RoiO;
  int BPJ;
  int t;
  PyArrayObject *lon, *lat, *rol, *d, *xd, *yd, *zd;
  PyArrayObject *cdelt1, *cdelt2, *crpix1, *crpix2;
  PyObject *ocdelt1, *ocdelt2, *ocdelt3;
  float cube_cdelt1, cube_cdelt2, cube_cdelt3;
  PyObject *ocrpix1, *ocrpix2, *ocrpix3;
  float cube_crpix1, cube_crpix2, cube_crpix3;
  /*test*/

  /*integers : dimension of the input and output array */
  int nt, n1, n2, nx, ny, nz;

  /* Parse tuples separately since args will differ between C fcns */
  if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &data,
			&PyArray_Type, &map, &BPJ))
    return NULL;
  /*Raise errors if input matrix is missing*/
  if (NULL == data)
    return NULL;
  if (NULL == map)
    return NULL;
 
  /* Get data and map dimensions. */
  n1 = data->dimensions[0];
  n2 = data->dimensions[1];
  nt = data->dimensions[2];
  nx = map->dimensions[0];
  ny = map->dimensions[1];
  nz = map->dimensions[2];
  /* Get data header */
  header = PyObject_GetAttrString((PyObject*)data, "header");
  lon = (PyArrayObject*)PyDict_GetItemString(header, "LON");
  lat = (PyArrayObject*)PyDict_GetItemString(header, "LAT");
  rol = (PyArrayObject*)PyDict_GetItemString(header, "ROL");
  d = (PyArrayObject*)PyDict_GetItemString(header, "D");
  xd = (PyArrayObject*)PyDict_GetItemString(header, "XD");
  yd = (PyArrayObject*)PyDict_GetItemString(header, "YD");
  zd = (PyArrayObject*)PyDict_GetItemString(header, "ZD");
  cdelt1 = (PyArrayObject*)PyDict_GetItemString(header, "CDELT1");
  cdelt2 = (PyArrayObject*)PyDict_GetItemString(header, "CDELT2");
  crpix1 = (PyArrayObject*)PyDict_GetItemString(header, "CRPIX1");
  crpix2 = (PyArrayObject*)PyDict_GetItemString(header, "CRPIX2");
  /* Region of interest of the object */
  cube_header = PyObject_GetAttrString((PyObject*)map, "header");
  RoiO.n[0] = nx;
  RoiO.n[1] = ny;
  RoiO.n[2] = nz;
  ocdelt1 = (PyObject*)PyDict_GetItemString(cube_header, "CDELT1");
  ocdelt2 = (PyObject*)PyDict_GetItemString(cube_header, "CDELT2");
  ocdelt3 = (PyObject*)PyDict_GetItemString(cube_header, "CDELT3");
  RoiO.p[0] = (float)PyFloat_AsDouble(ocdelt1);
  RoiO.p[1] = (float)PyFloat_AsDouble(ocdelt2);
  RoiO.p[2] = (float)PyFloat_AsDouble(ocdelt3);
  RoiO.d[0] = RoiO.p[0] * nx;
  RoiO.d[1] = RoiO.p[1] * ny;
  RoiO.d[2] = RoiO.p[2] * nz;
  ocrpix1 = (PyObject*)PyDict_GetItemString(cube_header, "CRPIX1");
  ocrpix2 = (PyObject*)PyDict_GetItemString(cube_header, "CRPIX2");
  ocrpix3 = (PyObject*)PyDict_GetItemString(cube_header, "CRPIX3");
  cube_crpix1 = (float)PyFloat_AsDouble(ocrpix1);
  cube_crpix2 = (float)PyFloat_AsDouble(ocrpix2);
  cube_crpix3 = (float)PyFloat_AsDouble(ocrpix3);
  RoiO.min[0] = - cube_crpix1 * RoiO.p[0];
  RoiO.min[1] = - cube_crpix2 * RoiO.p[1];
  RoiO.min[2] = - cube_crpix3 * RoiO.p[2];
  RoiO.max[0] = RoiO.min[0] + RoiO.d[0];
  RoiO.max[1] = RoiO.min[1] + RoiO.d[1];
  RoiO.max[2] = RoiO.min[2] + RoiO.d[2];
  /*printf("test\n");*/
  /*printf("%f\n", RoiO.p[0]);*/
  /* Loop on the time / image dimension */
  #pragma omp parallel shared(RoiO, data, map, BPJ) private(t, orbit, detector)
  #pragma omp for
  for(t = 0 ; t < nt ; t++){
    /* define orbit of current image */
    orbit.lon = FIND1(lon, t);
    orbit.lat = FIND1(lat, t);
    orbit.rol = FIND1(rol, t);
    orbit.d = FIND1(d, t);
    orbit.M[0] = FIND1(xd, t);
    orbit.M[1] = FIND1(yd, t);
    orbit.M[2] = FIND1(zd, t);
    /* define the detector of current image */
    detector.p1 = FIND1(cdelt1, t);
    detector.p2 = FIND1(cdelt2, t);
    detector.s1 = FIND1(crpix1, t);
    detector.s2 = FIND1(crpix2, t);
    detector.n1 = n1;
    detector.n2 = n2;
    /* Siddon for each time index */
    Siddon(data, t, map, orbit, RoiO, detector, BPJ);
  }
  Py_RETURN_NONE;
}


#define isNaN(x) ((x) != (x))
#define SQ(x) ((x) * (x))

double min3(double,double,double);
double max3(double,double,double);
int signe(double);
void Compare(double*,double*,double,double);

int Siddon(PyArrayObject * data,
	   int t,
	   PyArrayObject * cube,
	   orbit orbit,
	   RoiO RoiO,
	   detector detector,
	   int BPJ)
{
  /* declarations */
  /* loop incremented integers*/
  int i, j /*,k*/;
  /* to store constants defining the ray direction */
  /* lambda : latitude, longitude */
  double lambda, gamma;
  /* normalized direction vector in image referentiel */
  double u2[3];
  /* in solar referentiel */
  double u0[3];
  /* rotation matrix from image to solar referentiel */
  double R[3][3];
  /* distance of the current voxel to the detector   */
  double ac; 
  /* array containing the distances to the 6 faces of the volume*/
  double a1[3];
  double an[3];
  /* minimum of the distance array and it subscript */
  double amin, amax;
  /* coordinates of the initial and final points */
  double e[3];
  /* intersections avec les differentes faces */
  double axmin, axmax, aymin, aymax, azmin, azmax;
  /* subscripts of the current voxel */
  int iv[3];
  /* voxel initial */
  int ie[3];
  /* distances to the next intersection with a x,y or z constant
     plan of the grid */
  double p[3];
  /* current distances to the next intersection with a x,y or z 
     constant plan of the grid */
  double D[3];
  /* to discriminate between increasing and decreasing of voxel 
     subscripts*/
  int iupdate, jupdate, kupdate, inext, jnext, knext, itemp, jtemp, ktemp;
  /* distance to Sun center */
  double d;
  
  /* eq 10 with roll angle */
  rotation_matrix(orbit, R);
  /* loops on angles (detectors pixels) */
  for(i = 0 ; i < detector.n1 ; i++)
  {
    gamma = (i - detector.s1) * detector.p1; /*eq (9) */ 
    for(j = 0 ; j < detector.n2 ; j++)
    {
      /* skip computation if the value is a NaN */
      if( (!BPJ) || (isNaN(FIND3(data, i, j, t)) == 0 ) )
      {
	lambda = (j - detector.s2) * detector.p2; /*eq (9) */
	define_unit_vector(lambda, gamma, u2);
	apply_rotation(R, u2, u0);
	/* distances between 2 intersections of each kind */
	/* impact point determination */
	/* distances to faces */
	if(u0[0] == 0){
	  p[0] = INF;
	  a1[0] = INF;
	  an[0] = INF;}
	else{
	  p[0] = RoiO.p[0]/u0[0];
	  a1[0] = (RoiO.min[0] - orbit.M[0])/u0[0];
	  an[0] = (RoiO.max[0] - orbit.M[0])/u0[0];}
	if(u0[1] == 0){
	  p[1] = INF;
	  a1[1] = INF;
	  an[1] = INF;}
	else{
	  p[1] = RoiO.p[1]/u0[1];
	  a1[1] = (RoiO.min[1] - orbit.M[1])/u0[1];
	  an[1] = (RoiO.max[1] - orbit.M[1])/u0[1];}
	if(u0[2] == 0){
	  p[2] = INF;
	  a1[2] = INF;
	  an[2] = INF;}
	else{
	  p[2] = RoiO.p[2]/u0[2];
	  a1[2] = (RoiO.min[2] - orbit.M[2])/u0[2];
	  an[2] = (RoiO.max[2] - orbit.M[2])/u0[2];}
 
	Compare(&axmin,&axmax,a1[0],an[0]);
	Compare(&aymin,&aymax,a1[1],an[1]);
	Compare(&azmin,&azmax,a1[2],an[2]);

	amin = max3(axmin,aymin,azmin);
	amax = min3(axmax,aymax,azmax);
	
	if(amin < amax)
	{
	  /* initial and final points in cartesian coordinates */
	  e[0] = orbit.M[0] + amin * u0[0];
	  e[1] = orbit.M[1] + amin * u0[1];
	  e[2] = orbit.M[2] + amin * u0[2];
	  
	  iupdate = signe(u0[0]);
	  jupdate = signe(u0[1]);
	  kupdate = signe(u0[2]);
	  
	  /*  initial intersection*/ 
	  itemp = (int)( (e[0] - RoiO.min[0]) / RoiO.p[0]);
	  jtemp = (int)( (e[1] - RoiO.min[1]) / RoiO.p[1]);
	  ktemp = (int)( (e[2] - RoiO.min[2]) / RoiO.p[2]);
	  /* initial voxel of each kind */
	  ie[0] = itemp - (int)( (e[0] - RoiO.min[0]) / RoiO.d[0]);
	  ie[1] = jtemp - (int)( (e[1] - RoiO.min[1]) / RoiO.d[1]);
	  ie[2] = ktemp - (int)( (e[2] - RoiO.min[2]) / RoiO.d[2]);
	  
	  /* next intersection of each kind */
	  if(iupdate == 1)
	    inext = ie[0] + 1;
	  else if(iupdate == -1)
	    inext = ie[0];
	  else
	    inext = INF*RoiO.n[0];
	  if(jupdate == 1)
	    jnext = ie[1] + 1;
	  else if(jupdate == -1)
	    jnext = ie[1];
	  else
	    jnext = INF*RoiO.n[1];
	  if(kupdate == 1)
	    knext = ie[2] + 1;
	  else if(kupdate == -1)
	    knext = ie[2];
	  else
	    knext = INF*RoiO.n[2];
	  
	  D[0] = inext * p[0] + a1[0] - amin;
	  D[1] = jnext * p[1] + a1[1] - amin;
	  D[2] = knext * p[2] + a1[2] - amin;
	   
	  /*boucle initilisation */
	  ac = amin;
	  d = distance_to_center(orbit, u0, ac);
	  iv[0] = ie[0];
	  iv[1] = ie[1];
	  iv[2] = ie[2];

	  while( (iv[0] >= 0) && (iv[0] <RoiO.n[0]) && (iv[1] >= 0) && (iv[1] < RoiO.n[1] ) && (iv[2] >= 0 ) && (iv[2] < RoiO.n[2]) && (d > 1 ))
	  {
	    /* discriminate intersection with x,y and z = cte plan */
	    if((D[0]<=D[1])&&(D[0]<=D[2]))
	    {
	      {
		ac += D[0];
		d = distance_to_center(orbit, u0, ac);
		/* projection/backprojection*/
		if(!BPJ)
		  FIND3(data, i, j, t) += D[0] * FIND3(cube, iv[0], iv[1], iv[2]);
		else
		  FIND3(cube, iv[0], iv[1], iv[2]) += D[0] * FIND3(data, i, j, t);
		/* update voxel subscript */
		iv[0] += iupdate;
		/* update distances to next intersections*/
		D[1] -= D[0];
		D[2] -= D[0];
		D[0] = fabs(p[0]);
	      }
	    }
	    else if((D[1]<D[0])&&(D[1]<=D[2]))
	    {
	      ac += D[1];
	      d = distance_to_center(orbit, u0, ac);
	      /* projection/backprojection*/
	      if(!BPJ)
		FIND3(data, i, j, t) += D[1] * FIND3(cube, iv[0], iv[1], iv[2]);
	      else
		FIND3(cube, iv[0], iv[1], iv[2]) += D[1] * FIND3(data, i, j, t);
	      /* update voxel subscript */
	      iv[1] += jupdate;
	      /* update distances to next intersections*/
	      D[0] -= D[1];
	      D[2] -= D[1];
	      D[1] = fabs(p[1]);
	    }
	    else if((D[2]<D[0])&&(D[2]<D[1]))
	    {
	      ac += D[2];
	      d = distance_to_center(orbit, u0, ac);
	      /* projection/backprojection*/
	      if(!BPJ)
		FIND3(data, i, j, t) += D[2] * FIND3(cube, iv[0], iv[1], iv[2]);
	      else
		FIND3(cube, iv[0], iv[1], iv[2]) += D[2] * FIND3(data, i, j, t);
	      /* update voxel subscript */
	      iv[2] += kupdate;
	      /* update distances to next intersections*/
	      D[0] -= D[2];
	      D[1] -= D[2];
	      D[2] = fabs(p[2]);
	    }
	  }
	}
      }
    }
  }
  return 0;
}

double min3(double x, double y, double z)
{
  if((x < y)&(x < z))
    return x;
  else if ((y < x)&(y < z))
    return y;
  else
    return z;
}

double max3(double x, double y, double z)
{
  if((x > y)&(x > z))
    return x;
  else if ((y > x)&(y > z))
    return y;
  else
    return z;
}

int signe(double x)
{
  if (x>0)
    return 1;
  else if(x<0)
    return -1;
  else
    return 0;
}

void Compare(double * pumin, double * pumax, double u1, double u2)
{
  if (u1 > u2)
  {
    (*pumin) = u2;
    (*pumax) = u1;
  }
  else
  {
    (*pumin) = u1;
    (*pumax) = u2;    
  }
}

int rotation_matrix(orbit orbit, double R[3][3])
{
  /* to define the rotation matrix*/
  double cosln,sinln,coslt,sinlt,cosrl,sinrl;

  cosln = cos(orbit.lon);
  sinln = sin(orbit.lon);
  coslt = cos(orbit.lat);
  sinlt = sin(orbit.lat);
  cosrl = cos(orbit.rol);
  sinrl = sin(orbit.rol);

  R[0][0] = - cosln * coslt;
  R[0][1] = - sinln * cosrl - cosln * sinlt * sinrl;
  R[0][2] =   sinln * sinrl - cosln * sinlt * cosrl;
  R[1][0] = - sinln * coslt;
  R[1][1] =   cosln * cosrl - sinln * sinlt * sinrl;
  R[1][2] = - cosln * sinrl - sinln * sinlt * cosrl;
  R[2][0] = - sinlt;
  R[2][1] =   coslt * sinrl;
  R[2][2] =   coslt * cosrl;

  return 0;
}

int define_unit_vector(double lambda, double gamma, double u2[3])
{
  u2[0] = cos(lambda)*cos(gamma);
  u2[1] = cos(lambda)*sin(gamma);
  u2[2] = sin(lambda);

  return 0;
}

int apply_rotation(double R[3][3], double u2[3], double u0[3])
{
  u0[0] = R[0][0] * u2[0] + R[0][1] * u2[1] + R[0][2] * u2[2];
  u0[1] = R[1][0] * u2[0] + R[1][1] * u2[1] + R[1][2] * u2[2];
  u0[2] = R[2][0] * u2[0] + R[2][1] * u2[1] + R[2][2] * u2[2];

  return 0;
}

double distance_to_center(orbit orbit, double u0[3], double ac)
{
  double d;
  d = SQ(orbit.M[0] + ac * u0[0]) + SQ(orbit.M[1] + ac * u0[1]) + SQ(orbit.M[2] + ac * u0[2]);
  return d;
}
