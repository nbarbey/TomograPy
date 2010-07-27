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
  RoiO.nx = nx;
  RoiO.ny = ny;
  RoiO.nz = nz;
  ocdelt1 = (PyObject*)PyDict_GetItemString(cube_header, "CDELT1");
  ocdelt2 = (PyObject*)PyDict_GetItemString(cube_header, "CDELT2");
  ocdelt3 = (PyObject*)PyDict_GetItemString(cube_header, "CDELT3");
  RoiO.px = (float)PyFloat_AsDouble(ocdelt1);
  RoiO.py = (float)PyFloat_AsDouble(ocdelt2);
  RoiO.pz = (float)PyFloat_AsDouble(ocdelt3);
  RoiO.dx = RoiO.px * nx;
  RoiO.dy = RoiO.py * ny;
  RoiO.dz = RoiO.pz * nz;
  ocrpix1 = (PyObject*)PyDict_GetItemString(cube_header, "CRPIX1");
  ocrpix2 = (PyObject*)PyDict_GetItemString(cube_header, "CRPIX2");
  ocrpix3 = (PyObject*)PyDict_GetItemString(cube_header, "CRPIX3");
  cube_crpix1 = (float)PyFloat_AsDouble(ocrpix1);
  cube_crpix2 = (float)PyFloat_AsDouble(ocrpix2);
  cube_crpix3 = (float)PyFloat_AsDouble(ocrpix3);
  RoiO.xmin = - cube_crpix1 * RoiO.px;
  RoiO.ymin = - cube_crpix2 * RoiO.py;
  RoiO.zmin = - cube_crpix3 * RoiO.pz;
  RoiO.xmax = RoiO.xmin + RoiO.dx;
  RoiO.ymax = RoiO.ymin + RoiO.dy;
  RoiO.zmax = RoiO.zmin + RoiO.dz;
  /*printf("test\n");*/
  /*printf("%f\n", RoiO.px);*/
  /* Loop on the time / image dimension */
  #pragma omp parallel shared(RoiO, data, map, BPJ) private(t, orbit, detector)
  #pragma omp for
  for(t = 0 ; t < nt ; t++){
    /* define orbit of current image */
    orbit.lon = FIND1(lon, t);
    orbit.lat = FIND1(lat, t);
    orbit.rol = FIND1(rol, t);
    orbit.d = FIND1(d, t);
    orbit.xd = FIND1(xd, t);
    orbit.yd = FIND1(yd, t);
    orbit.zd = FIND1(zd, t);
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
  /* lambda : latitude of the current line */
  double lambda;
  /* gamma : longitude of the current line */
  double gamma;
  /* normalized direction vector in image referentiel */
  double u2[3];
  /* in solar referentiel */
  double u0[3];
    /* rotation matrix from image to solar referentiel */
  double R[3][3];
  /* distance of the current voxel to the detector   */
  double ac; 
  /* array containing the distances to the 6 faces of the volume*/
  double ax1,axn,ay1,ayn,az1,azn;
  /* minimum of the distance array and it subscript */
  double amin,amax;
  /* coordinates of the initial and final points */
  double xe,ye,ze;
  /* intersections avec les differentes faces */
  double axmin,axmax,aymin,aymax,azmin,azmax;  	      
  /* subscripts of the current voxel */
  int iv,jv,kv;
  /* voxel initial */
  int ie,je,ke;
  /* distances to the next intersection with a x,y or z constant
     plan of the grid */
  double dx,dy,dz;
  /* current distances to the next intersection with a x,y or z 
     constant plan of the grid */
  double Dx,Dy,Dz;
  /* to discriminate between increasing and decreasing of voxel 
     subscripts*/
  int iupdate,jupdate,kupdate,inext,jnext,knext,itemp,jtemp,ktemp;
  /* to define the rotation matrix*/
  double cosln,sinln,coslt,sinlt,cosrl,sinrl;
  /* distance to Sun center */
  double d;
  
  /* eq 10 with roll angle */
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

  /* loops on angles (detectors pixels) */
  for(i = 0 ; i < detector.n1 ; i++)
  {
    gamma = (i-detector.s1)*detector.p1; /*eq (9) */ 
    for(j = 0 ; j < detector.n2 ; j++)
    {
      /* ne calcul rien si la valeure est NaN */
      if( (!BPJ) || (isNaN(FIND3(data, i, j, t)) == 0 ) )
      {
	lambda = (j-detector.s2)*detector.p2; /*eq (9) */
	
	u2[0] = cos(lambda)*cos(gamma);
	u2[1] = cos(lambda)*sin(gamma);
	u2[2] = sin(lambda);
	
	u0[0] = R[0][0] * u2[0] + R[0][1] * u2[1] + R[0][2] * u2[2];
	u0[1] = R[1][0] * u2[0] + R[1][1] * u2[1] + R[1][2] * u2[2];
	u0[2] = R[2][0] * u2[0] + R[2][1] * u2[1] + R[2][2] * u2[2];

	/* distances between 2 intersections of each kind */
	/* impact point determination */
	/* distances to faces */
	if(u0[0] == 0){
	  dx = INF;
	  ax1 = INF;
	  axn = INF;}
	else{
	  dx = RoiO.px/u0[0];
	  ax1 = (RoiO.xmin - orbit.xd)/u0[0];
	  axn = (RoiO.xmax - orbit.xd)/u0[0];}
	if(u0[1] == 0){
	  dy = INF;
	  ay1 = INF;
	  ayn = INF;}
	else{
	  dy = RoiO.py/u0[1];
	  ay1 = (RoiO.ymin - orbit.yd)/u0[1];
	  ayn = (RoiO.ymax - orbit.yd)/u0[1];}
	if(u0[2] == 0){
	  dz = INF;
	  az1 = INF;
	  azn = INF;}
	else{
	  dz = RoiO.pz/u0[2];
	  az1 = (RoiO.zmin - orbit.zd)/u0[2];
	  azn = (RoiO.zmax - orbit.zd)/u0[2];}
 
	Compare(&axmin,&axmax,ax1,axn);
	Compare(&aymin,&aymax,ay1,ayn);
	Compare(&azmin,&azmax,az1,azn);

	amin = max3(axmin,aymin,azmin);
	amax = min3(axmax,aymax,azmax);
	
	if(amin < amax)
	{
	  /* initial and final points in cartesian coordinates */
	  xe = orbit.xd + amin * u0[0];
	  ye = orbit.yd + amin * u0[1];
	  ze = orbit.zd + amin * u0[2];
	  
	  iupdate = signe(u0[0]);
	  jupdate = signe(u0[1]);
	  kupdate = signe(u0[2]);
	  
	  /*  initial intersection*/ 
	  itemp = (int)( (xe - RoiO.xmin) / RoiO.px);
	  jtemp = (int)( (ye - RoiO.ymin) / RoiO.py);
	  ktemp = (int)( (ze - RoiO.zmin) / RoiO.pz);
	  /* initial voxel of each kind */
	  ie = itemp - (int)( (xe - RoiO.xmin) / RoiO.dx);
	  je = jtemp - (int)( (ye - RoiO.ymin) / RoiO.dy);
	  ke = ktemp - (int)( (ze - RoiO.zmin) / RoiO.dz);
	  
	  /* next intersection of each kind */
	  if(iupdate == 1)
	    inext = ie + 1;
	  else if(iupdate == -1)
	    inext = ie;
	  else
	    inext = INF*RoiO.nx;
	  if(jupdate == 1)
	    jnext = je + 1;
	  else if(jupdate == -1)
	    jnext = je;
	  else
	    jnext = INF*RoiO.ny;
	  if(kupdate == 1)
	    knext = ke + 1;
	  else if(kupdate == -1)
	    knext = ke;
	  else
	    knext = INF*RoiO.nz;
	  
	  Dx = inext * dx + ax1 - amin;
	  Dy = jnext * dy + ay1 - amin;
	  Dz = knext * dz + az1 - amin;
	   
	  /*boucle initilisation */
	  ac = amin;
	  d = SQ( orbit.xd + ac * u0[0]) + SQ( orbit.yd + ac * u0[1]) + SQ( orbit.zd + ac * u0[2]);
	  iv = ie;
	  jv = je;
	  kv = ke;

	  while( (iv >= 0) && (iv <RoiO.nx) && (jv >= 0) && (jv < RoiO.ny ) && (kv >= 0 ) && (kv < RoiO.nz) && (d > 1 ))
	  {
	    /* discriminate intersection with x,y and z = cte plan */
	    if((Dx<=Dy)&&(Dx<=Dz))
	    {
	      {
		ac += Dx;
		d = SQ( orbit.xd + ac * u0[0]) + SQ( orbit.yd + ac * u0[1]) + SQ( orbit.zd + ac * u0[2]);
		/* projection/backprojection*/
		if(!BPJ)
		  FIND3(data, i, j, t) += Dx * FIND3(cube, iv, jv, kv);
		else
		  FIND3(cube, iv, jv, kv) += Dx * FIND3(data, i, j, t);
		/* update voxel subscript */
		iv += iupdate;
		/* update distances to next intersections*/
		Dy -= Dx;
		Dz -= Dx;
		Dx = fabs(dx);
	      }
	    }
	    else if((Dy<Dx)&&(Dy<=Dz))
	    {
	      ac += Dy;
	      d = SQ( orbit.xd + ac * u0[0]) + SQ( orbit.yd + ac * u0[1]) + SQ( orbit.zd + ac * u0[2]);
	      /* projection/backprojection*/
	      if(!BPJ)
		FIND3(data, i, j, t) += Dy * FIND3(cube, iv, jv, kv);
	      else
		FIND3(cube, iv, jv, kv) += Dy * FIND3(data, i, j, t);
	      /* update voxel subscript */
	      jv += jupdate;
	      /* update distances to next intersections*/
	      Dx -= Dy;
	      Dz -= Dy;
	      Dy = fabs(dy);
	    }
	    else if((Dz<Dx)&&(Dz<Dy))
	    {
	      ac += Dz;
	      d = SQ( orbit.xd + ac * u0[0]) + SQ( orbit.yd + ac * u0[1]) + SQ( orbit.zd + ac * u0[2]);
	      /* projection/backprojection*/
	      if(!BPJ)
		FIND3(data, i, j, t) += Dz * FIND3(cube, iv, jv, kv);
	      else
		FIND3(cube, iv, jv, kv) += Dz * FIND3(data, i, j, t);
	      /* update voxel subscript */
	      kv += kupdate;
	      /* update distances to next intersections*/
	      Dx -= Dz;
	      Dy -= Dz;
	      Dz = fabs(dz);
	    }
	  }
	}
      }
    }
  }
  return 0;
}

double min3(double x,double y,double z)
{
  if((x < y)&(x < z))
    return x;
  else if ((y < x)&(y < z))
    return y;
  else
    return z;
}

double max3(double x,double y,double z)
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

void Compare(double * pumin,double * pumax, double u1, double u2)
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
