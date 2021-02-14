﻿using System;

namespace OpenCvSharp
{
    /// <summary>
    /// A class which has a pointer of OpenCV structure
    /// </summary>
    public abstract class CvObject : ICvPtrHolder
    {
        /// <summary>
        /// Data pointer
        /// </summary>
        protected IntPtr ptr;

        /// <summary>
        /// Default constructor
        /// </summary>
        protected CvObject()
        {
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="ptr"></param>
        protected CvObject(IntPtr ptr)
        {
            this.ptr = ptr;
        }

        /// <summary>
        /// Native pointer of OpenCV structure
        /// </summary>
        public IntPtr CvPtr
        {
            get { return ptr; }
        }
    }
}