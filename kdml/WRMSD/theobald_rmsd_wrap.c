// This file is part of MSMBuilder.
//
// Copyright 2011 Stanford University
//
// MSMBuilder is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//


#include "Python.h"
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <stdio.h>
#include "theobald_rmsd.h"
#include <omp.h>

extern PyObject *corresponding(PyObject *self, PyObject *args) {
  PyArrayObject *A_, *B_, *GA_, *GB_, *deviations_;
  const float *A, *B, *GA, *GB;
  float *deviations;
  int nrealatoms, npaddedatoms, rowstride, nframes, truestride;
  int parallel;
  
   if (!PyArg_ParseTuple(args, "iiiiO!O!O!O!O!",
     &nrealatoms, &npaddedatoms, &rowstride, &parallel,
     &PyArray_Type, &A_, &PyArray_Type, &B_,
     &PyArray_Type, &GA_, &PyArray_Type, &GB_,
     &PyArray_Type, &deviations_)) {
       return 0;
  }
  else {
    A = (const float*)A_->data;
    B = (const float*)B_->data;
    GA = (const real*)GA_->data;
    GB = (const real*)GB_->data;
    deviations = (float*)deviations_->data;
    
    nframes = A_->dimensions[0];
  }
  
  truestride = npaddedatoms * 3;
  
  #pragma omp parallel for if (parallel > 0)
  for (int i = 0; i < nframes; i++) {
    aligned_deviation(nrealatoms, npaddedatoms, rowstride, (A+i*truestride),
      (B+i*truestride), GA[i], GB[i], deviations + i*nrealatoms);
  }

  return Py_BuildValue("d", 0.0);
};

extern PyObject *one_to_all(PyObject *self, PyObject *args) {
  PyArrayObject *Aframe_, *B_, *GB_, *deviations_;
  const float *Aframe, *B, *GB;
  float *deviations;
  float GAframe;
  int nrealatoms, npaddedatoms, rowstride, nframes, truestride;
  int parallel;
  
  
   if (!PyArg_ParseTuple(args, "iiiifO!O!O!O!",
     &nrealatoms, &npaddedatoms, &rowstride, &parallel, &GAframe,
     &PyArray_Type, &Aframe_, &PyArray_Type, &B_,
     &PyArray_Type, &GB_, &PyArray_Type, &deviations_)) {
       return 0;
  }
  else {
    Aframe = (const float*)Aframe_->data;
    B = (const float*)B_->data;
    GB = (const real*)GB_->data;
    deviations = (float*)deviations_->data;
    
    nframes = B_->dimensions[0];
  }
  
  
  truestride = npaddedatoms * 3;
  
  #pragma omp parallel for if (parallel > 0)
  for (int i = 0; i < nframes; i++) {
    aligned_deviation(nrealatoms, npaddedatoms, rowstride, Aframe,
      (B+i*truestride), GAframe, GB[i], deviations + i*nrealatoms);
  }

  return Py_BuildValue("d", 0.0);
};

extern PyObject *one_to_many(PyObject *self, PyObject *args) {
  PyArrayObject *Aframe_, *B_, *GB_, *indicesB_, *deviations_;
  const float *Aframe, *B, *GB;
  const int *indicesB;
  float *deviations;
  float GAframe;
  int nrealatoms, npaddedatoms, rowstride, n_indicesB;
  int indx, truestride;
  int parallel;
  
   if (!PyArg_ParseTuple(args, "iiiifO!O!O!O!O!",
     &nrealatoms, &npaddedatoms, &rowstride, &parallel, &GAframe,
     &PyArray_Type, &Aframe_, &PyArray_Type, &B_,
     &PyArray_Type, &GB_, &PyArray_Type, &indicesB_,
     &PyArray_Type, &deviations_)) {
       return 0;
  }
  else {
    Aframe = (const float*)Aframe_->data;
    B = (const float*)B_->data;
    GB = (const real*)GB_->data;
    indicesB = (const int*)indicesB_->data;
    deviations = (float*)deviations_->data;
    
    n_indicesB = indicesB_->dimensions[0];
  }
  
  truestride = npaddedatoms * 3;
  
  #pragma omp parallel for if (parallel > 0)
  for (int i = 0; i < n_indicesB; i++) {
      indx = indicesB[i];
      aligned_deviation(nrealatoms, npaddedatoms, rowstride, Aframe,
      (B+indx*truestride), GAframe, GB[indx], deviations + i*nrealatoms);
  }
  
  return Py_BuildValue("d", 0.0);
};

static PyMethodDef _WRMSD_methods[] = {
  {"corresponding", corresponding, METH_VARARGS},
  {"one_to_all", one_to_all, METH_VARARGS},
  {"one_to_many", one_to_many, METH_VARARGS},
  {NULL, NULL} /* Sentinel -- marks the end of this structure*/
};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_WRMSD",
    NULL,
    -1,
    _WRMSD_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyObject *PyInit__WRMSD(void)
{
    PyObject *m;

    m = PyModule_Create(&moduledef);
    import_array();

    return m;
}
#else
PyMODINIT_FUNC init_WRMSD(void)
{
  (void) Py_InitModule("_WRMSD", _WRMSD_methods);
  import_array();  // Must be present for NumPy.  Called first after above line.
}
#endif
