/*
 * -*- coding: utf-8 -*-
 * Copyright (c) 2009.
 *
 * SMHI,
 * Folkborgsvägen 1,
 * Norrköping, 
 * Sweden
 *
 * Author(s):
 *
 *   Martin Raspaud <martin.raspaud@smhi.se>
 *   Adam Dybbroe <adam.dybbroe@smhi.se>
 *
 * This file is part of the MPPP.
 *
 * MPPP is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * MPPP is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with MPPP.  If not, see <http://www.gnu.org/licenses/>.
 */


/*
 * This is the python module interface to the get_channel utility,
 * which allows to grab a Seviri channel from MSG data (using the MSG
 * library).
 * 
 */

#include <stdbool.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <string.h>
#include <error.h>
#include <NWCLIB/libnwc.h>
#include <NWCLIB/sevext.h>

int
channel_number(char * channel_string)
{
  int channel=-1;

  if(strcmp(channel_string,"VIS06")==0 || atoi(channel_string)==1)
    channel=VIS06;
  else if(strcmp(channel_string,"VIS08")==0 || atoi(channel_string)==2)
    channel=VIS08;
  else if(strcmp(channel_string,"IR16")==0 || atoi(channel_string)==3)
    channel=IR16;
  else if(strcmp(channel_string,"IR39")==0 || atoi(channel_string)==4)
    channel=IR39;
  else if(strcmp(channel_string,"WV62")==0 || atoi(channel_string)==5)
    channel=WV62;
  else if(strcmp(channel_string,"WV73")==0 || atoi(channel_string)==6)
    channel=WV73;
  else if(strcmp(channel_string,"IR87")==0 || atoi(channel_string)==7)
    channel=IR87;
  else if(strcmp(channel_string,"IR97")==0 || atoi(channel_string)==8)
    channel=IR97;
  else if(strcmp(channel_string,"IR108")==0 || atoi(channel_string)==9)
    channel=IR108;
  else if(strcmp(channel_string,"IR120")==0 || atoi(channel_string)==10)
    channel=IR120;
  else if(strcmp(channel_string,"IR134")==0 || atoi(channel_string)==11)
    channel=IR134;
  else if(strcmp(channel_string,"HRVIS")==0 || atoi(channel_string)==12)
    channel=HRVIS;
  else
    {
      char error_msg[256];
      sprintf(error_msg,"%s: Channel name not recognized.",channel_string);
      PyErr_SetString(PyExc_KeyError,error_msg);
      return -1;
    }


  return channel;

}

/*
 * channel_name_string
 *
 * Sets a channel name (string) from its number.
 * 
 */

void
channel_name_string(int channel, char * channel_string)
{
  if(channel == VIS06)
    sprintf(channel_string,"VIS06");
  if(channel == VIS08)
    sprintf(channel_string,"VIS08");
  if(channel == IR16)
    sprintf(channel_string,"IR16");
  if(channel == IR39)
    sprintf(channel_string,"IR39");
  if(channel == WV62)
    sprintf(channel_string,"WV62");
  if(channel == WV73)
    sprintf(channel_string,"WV73");
  if(channel == IR87)
    sprintf(channel_string,"IR87");
  if(channel == IR97)
    sprintf(channel_string,"IR97");
  if(channel == IR108)
    sprintf(channel_string,"IR108");
  if(channel == IR120)
    sprintf(channel_string,"IR120");
  if(channel == IR134)
    sprintf(channel_string,"IR134");
  if(channel == HRVIS)
    sprintf(channel_string,"HRVIS");

}

void
make_mask(PyArrayObject * in, PyArrayObject * out, npy_intp * dims)
{
  Py_ssize_t i,j;
  for(i=0;i<dims[0];i++)
    for(j=0;j<dims[1];j++)
      {
	*((npy_bool *)PyArray_GETPTR2(out,i,j))=
          (npy_bool)(*((npy_float *)PyArray_GETPTR2(in,i,j)) == 
                     SEVIRI_MISSING_VALUE);
      }
  
}

void
copy_to_2Dfloat_pyarray(Float_32 ** in, PyArrayObject * out, npy_intp * dims)
{
  Py_ssize_t i,j;
  for(i=0;i<dims[0];i++)
    for(j=0;j<dims[1];j++)
      {
	*((npy_float *)PyArray_GETPTR2(out,i,j))=(npy_float)in[i][j];
      }
  
}

PyObject *
SimpleNewFrom2DData(int nd, npy_intp* dims, int typenum, void* data)
{
  PyArrayObject * cal;
  cal = (PyArrayObject *)PyArray_SimpleNew(nd,dims,typenum);
  copy_to_2Dfloat_pyarray(data,cal,dims);
  return (PyObject *)cal;
}

static PyObject *
msg_get_channels(PyObject *dummy, PyObject *args)
{

  YYYYMMDDhhmm time_slot;
  char * c_time_slot;
  char * region_name;
  char region_file[128];
  unsigned char read_rad = true;
  PyObject * channels;
  int i, j;
  int ch;
  int channel;
  char channel_name[128];
  Band_mask bands;
  int int_bandmask = 0;
  int seviri_bands = 0;
  int hr_seviri_band = 0;

  unsigned char got_hr = false;
  unsigned char got_nonhr = false;

  Psing_region region;
  Psing_region hr_region;

  Seviri_struct seviri;
  Seviri_struct hr_seviri;

  npy_intp dims[2];
  npy_intp hr_dims[2];

  PyArrayObject *rad;
  PyArrayObject *cal;
  PyArrayObject *mask;

  PyObject * band;
  PyObject * chan_dict;

  // Parse args
  
  if (!PyArg_ParseTuple(args, "ssO|b", &c_time_slot, &region_name, &channels, &read_rad))
    {
      PyErr_SetString(PyExc_RuntimeError,"Impossible to parse arguments.");
      return NULL;
    }
  strcpy(time_slot,c_time_slot);

  if(!PyList_Check(channels))
    {
      PyErr_SetString(PyExc_RuntimeError,"Channels arg is not a list.");
      return NULL;
    }

  
  for(i = 0; i < NUM_BANDS; i++)
    {
      bands[i] = false;
    }

  for(i = 0; i < PyList_Size(channels); i++)
    {
      ch = channel_number(PyString_AsString(PyList_GetItem(channels,i)));
      if(ch == -1)
        return NULL;

      bands[ch] = true;

      int_bandmask |= (int)pow(2,ch);

      if(ch == HRVIS)
        got_hr = true;
      else
        got_nonhr = true;
    }

  // Init region

  sprintf(region_file,"safnwc_%s.cfg",region_name);

  if(got_hr)
    {
      if (SetRegion(&hr_region,region_file,HRV) > WARNING ) 
	{
          PyErr_SetString(PyExc_RuntimeError,"Could not initialize HR region.");
          return NULL;
	}     
      hr_dims[0] = hr_region.nb_lines;
      hr_dims[1] = hr_region.nb_cols;
    }

  if(got_nonhr)
    {
      if (SetRegion(&region,region_file,VIS_IR) > WARNING ) 
	{
          PyErr_SetString(PyExc_RuntimeError,"Could not initialize region.");
          return NULL;
	}
      dims[0] = region.nb_lines;
      dims[1] = region.nb_cols;
    }

  // Init seviri struct, hr_seviri if needed

  if(got_hr && (SevHRVInit(time_slot,hr_region,&hr_seviri)!=OK))
    {
      PyErr_SetString(PyExc_RuntimeError,"Cannot initialize HR Seviri reader.");
      return NULL;
    }
 
  if(got_nonhr && (SevInit(time_slot,region,bands,&seviri)!=OK))
    {
      PyErr_SetString(PyExc_RuntimeError,"Cannot initialize Seviri reader.");
      return NULL;
    }

  // Read channels from msg

  if(got_hr && (SevReadHRV(hr_region, &hr_seviri) > WARNING))
    {
      PyErr_SetString(PyExc_RuntimeError,"Cannot read HR Seviri data.");
      goto error;
    }
    
  if(got_nonhr && (SevRead(region, &seviri, bands) > WARNING))
    {
      PyErr_SetString(PyExc_RuntimeError,"Cannot read Seviri data.");
      goto error;
    }

  // Convert to radiances, reflectances, and bts.

  for(channel = 0; channel < NUM_BANDS; channel++)
     {
       if(bands[channel])
	 {
           if(channel == HRVIS)
             {
               if(read_rad)
                 SevCalibrate(&hr_seviri,channel);
               
               SevCalRefl(&hr_seviri,channel);
             }
           else
             {
               if(read_rad)
                 SevCalibrate(&seviri,channel);
               
               if(channel == VIS06 || channel == VIS08 || channel == IR16)
                 SevCalRefl(&seviri,channel);
               else
                 SevConvert(&seviri,channel);
             }
	 }
     }

  // Check the channels

  if(got_nonhr)
    seviri_bands = CheckSevBand(&seviri);
  if(got_hr)
    hr_seviri_band = CheckSevBand(&hr_seviri);

  int_bandmask ^= (hr_seviri_band | seviri_bands);

  j = 0;
  for(i = int_bandmask; i != 0; i >>= 1)
    {
      if(i & 1)
        bands[j] = false;
      j++;
    }

  // Append channels to a python dict

  if(!(chan_dict = PyDict_New()))
    {
      PyErr_SetString(PyExc_RuntimeError,"Cannot create a new dictionnary.");
      goto error;
    }
  
  for(channel = 0; channel < NUM_BANDS; channel++)
    if(bands[channel])
      {
        if(channel == HRVIS)
          {
            if(read_rad)
              rad = (PyArrayObject *)SimpleNewFrom2DData(2,hr_dims,NPY_FLOAT,
                                                         SevBand(hr_seviri,channel,RAD));
            else
              rad = (PyArrayObject *)PyArray_EMPTY(2, hr_dims, NPY_FLOAT,0);
            if(SevBand(hr_seviri,channel,REFL)!=NULL)
              {
                cal = (PyArrayObject *)SimpleNewFrom2DData(2,hr_dims,NPY_FLOAT,
                                                           SevBand(hr_seviri,channel,REFL));
              }
            else
              {
                cal = (PyArrayObject *)SimpleNewFrom2DData(2,hr_dims,NPY_FLOAT,
                                                           SevBand(hr_seviri,channel,BT));
              }
            mask = (PyArrayObject *)PyArray_SimpleNew(2, hr_dims, NPY_BOOL);
            make_mask(cal, mask, hr_dims);
          }
        else
          {
            if(read_rad)
              rad = (PyArrayObject *)SimpleNewFrom2DData(2,dims,NPY_FLOAT,
                                                         SevBand(seviri,channel,RAD));
            else
              rad = (PyArrayObject *)PyArray_EMPTY(2, dims, NPY_FLOAT,0);
            if(SevBand(seviri,channel,REFL)!=NULL)
              {
                cal = (PyArrayObject *)SimpleNewFrom2DData(2,dims,NPY_FLOAT,
                                                           SevBand(seviri,channel,REFL));
              }
            else
              {
                cal = (PyArrayObject *)SimpleNewFrom2DData(2,dims,NPY_FLOAT,
                                                           SevBand(seviri,channel,BT));
              }
            mask = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_BOOL);
            make_mask(cal, mask, dims);
          }
        band = Py_BuildValue("{s:N,s:N,s:N}",
                             "RAD", PyArray_Return(rad),
                             "CAL", PyArray_Return(cal),
                             "MASK",PyArray_Return(mask));
        channel_name_string(channel,channel_name);
        PyDict_SetItemString(chan_dict,channel_name,band);
      }

  // Cleanup

  if(got_hr)
    SevFree(hr_seviri);
  if(got_nonhr)
    SevFree(seviri);

  // Return the dict
  
  return chan_dict;

  // In case of error

 error:
  if(got_hr)
    SevFree(hr_seviri);
  if(got_nonhr)
    SevFree(seviri);

  return NULL;
  
  
}

static PyObject *
msg_lat_lon_from_region(PyObject *dummy, PyObject *args)
{

  Float_32 ** MSG_lat;
  Float_32 ** MSG_lon;
  
  Psing_region region;

  int channel;

  PyArrayObject *lat;
  PyArrayObject *lon;

  char * region_name;
  char region_file[128];
  char * channel_name;

  npy_intp dims[2];

  // Parse arguments
  
  if (!PyArg_ParseTuple(args, "ss", &region_name, &channel_name))
    {
      PyErr_SetString(PyExc_RuntimeError,"Impossible to parse arguments.");
      return NULL;
    }

  channel = channel_number(channel_name);
  if(channel==-1)
    {
      PyErr_SetString(PyExc_RuntimeError,"Unrecognized channel name.");
      return NULL;
    }

  // Init region and get lat lon.

  sprintf(region_file,"safnwc_%s.cfg",region_name);

  if(channel==HRVIS)
    {
      if (SetRegion(&region,region_file,HRV) > WARNING ) 
	{
          PyErr_SetString(PyExc_RuntimeError,"Could not initialize HR region.");
          return NULL;
	}
      GetLatLon(region, HRVIS, &MSG_lat, &MSG_lon);
    }
  else
    {
      if (SetRegion(&region,region_file,VIS_IR) > WARNING ) 
	{
          PyErr_SetString(PyExc_RuntimeError,"Could not initialize region.");
          return NULL;
	}
      GetLatLon(region, VIS_IR, &MSG_lat, &MSG_lon);
    }
  
  // Copy to numpy arrays

  dims[0] = region.nb_lines;
  dims[1] = region.nb_cols;

  lat = (PyArrayObject *)SimpleNewFrom2DData(2, dims, NPY_FLOAT, MSG_lat);
  lon = (PyArrayObject *)SimpleNewFrom2DData(2, dims, NPY_FLOAT, MSG_lon);

  // Cleanup

  FreeLatLon(region, MSG_lat, MSG_lon);

  // Return

  return Py_BuildValue("(N,N)",PyArray_Return(lat),PyArray_Return(lon));
}


static PyObject *
msg_missing_value()
{
  return Py_BuildValue("f",(float)SEVIRI_MISSING_VALUE);
}

static PyMethodDef MsgMethods[] = {
    {"get_channels",  msg_get_channels, METH_VARARGS,
     "Loads the data of a list of Seviri *channels* from MSG into numpy arrays,"
     " for given *time_slot* and *region_name*. If *read_rad* is true, the "
     "radiance values are loaded also.\n\nThe return value is presented as a "
     "dictionnary of dictionnaries. The first level of keys are the channel "
     "names, the second level contains RAD, CAL, MASK, respectively for the "
     "radiance data, calibrated data, and the mask of invalid data (true where"
     " data is invalid). RAD is None if the radiance is not loaded (default)."},
    {"lat_lon_from_region",  msg_lat_lon_from_region, METH_VARARGS,
     "Return a tuple of latitudes and longitudes 2D grids for a given "
     "*region_name* and a given *channel* (name)."},
    {"missing_value",  msg_missing_value, METH_VARARGS,
     "Returns the fill value for missing data."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initpy_msg(void)
{
    (void) Py_InitModule("py_msg", MsgMethods);
    import_array();
    Py_Initialize();
}

