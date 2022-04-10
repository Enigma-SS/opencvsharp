using System;
using System.Linq;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Runtime.InteropServices;
using OpenCvSharp.Internal;

namespace OpenCvSharp.Face
{
    /// <inheritdoc />
    /// <summary>
    /// Training and prediction must be done on grayscale images, use cvtColor to convert between the color spaces.
    /// -   **THE FISHERFACES METHOD MAKES THE ASSUMPTION, THAT THE TRAINING AND TEST IMAGES ARE OF EQUAL SIZE.
    ///     ** (caps-lock, because I got so many mails asking for this). You have to make sure your input data 
    ///       has the correct shape, else a meaningful exception is thrown.Use resize to resize the images.
    /// -   This model does not support updating.
    /// </summary>
    // ReSharper disable once InconsistentNaming
    public class FisherFaceRecognizer : BasicFaceRecognizer
    {
        /// <summary>
        ///
        /// </summary>
        private Ptr? recognizerPtr;

        /// <inheritdoc />
        ///  <summary>
        ///  </summary>
        protected FisherFaceRecognizer()
        {
            recognizerPtr = null;
            ptr = IntPtr.Zero;
        }
        
        /// <summary>
        /// Releases managed resources
        /// </summary>
        protected override void DisposeManaged()
        {
            recognizerPtr?.Dispose();
            recognizerPtr = null;
            base.DisposeManaged();
        }

        /// <summary>
        /// Training and prediction must be done on grayscale images, use cvtColor to convert between the color spaces.
        /// -   **THE FISHERFACES METHOD MAKES THE ASSUMPTION, THAT THE TRAINING AND TEST IMAGES ARE OF EQUAL SIZE.
        ///     ** (caps-lock, because I got so many mails asking for this). You have to make sure your input data 
        ///       has the correct shape, else a meaningful exception is thrown.Use resize to resize the images.
        /// -   This model does not support updating.
        /// </summary>
        /// <param name="numComponents">The number of components (read: Fisherfaces) kept for this Linear Discriminant Analysis 
        /// with the Fisherfaces criterion. It's useful to keep all components, that means the number of your classes c 
        /// (read: subjects, persons you want to recognize). If you leave this at the default (0) or set it 
        /// to a value less-equal 0 or greater (c-1), it will be set to the correct number (c-1) automatically.</param>
        /// <param name="threshold">The threshold applied in the prediction. If the distance to the nearest neighbor 
        /// is larger than the threshold, this method returns -1.</param>
        /// <returns></returns>
        public static FisherFaceRecognizer Create(int numComponents = 0, double threshold = double.MaxValue)
        {
            NativeMethods.HandleException(
                NativeMethods.face_FisherFaceRecognizer_create(numComponents, threshold, out var p));
            if (p == IntPtr.Zero)
                throw new OpenCvSharpException($"Invalid cv::Ptr<{nameof(FisherFaceRecognizer)}> pointer");
            var ptrObj = new Ptr(p);
            var detector = new FisherFaceRecognizer
            {
                recognizerPtr = ptrObj,
                ptr = ptrObj.Get()
            };
            return detector;
        }

        /// <summary>
        /// Gets a prediction from a FaceRecognizer.
        /// </summary>
        /// <param name="src"></param>
        /// <param name="topK"></param>
        /// <param name="confidence_threshold"></param>
        /// <returns></returns>
        public virtual OrderedDictionary PredictTopK(InputArray src, int topK, double disMax = 500.0, double simMax = 100.0)
        {
            ThrowIfDisposed();
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            src.ThrowIfDisposed();
            NativeMethods.HandleException(
                NativeMethods.face_FisherFaceRecognizer_predictTopK(ptr, src.CvPtr, out IntPtr _lbls, out IntPtr _dist, out int count));
            GC.KeepAlive(this);
            GC.KeepAlive(src);
            // check if value for topK is valid or not
            if (count < topK || topK == 0)
                throw new ArgumentException("Invalid value for topK, values must be <= # of classes and > 0");
            // assign memory the the output variables
            int[] labels = new int[count];
            double[] confidence = new double[count];
            // copy all results to the output arrays
            Marshal.Copy(_lbls, labels, 0, count);
            Marshal.Copy(_dist, confidence, 0, count);
            // create the dictionary
            Dictionary<int, double> all_data = new Dictionary<int, double>();
            // loop through the obtained results and return values
            for (int i = 0; i < count; i++)
            {
                all_data.Add(labels[i], Math.Abs(confidence[i]));
            }
            // setup output dictionary with sorted values
            OrderedDictionary orderedDictionary = new OrderedDictionary();
            foreach (KeyValuePair<int, double> item in all_data.OrderBy(key => key.Value))
            {
                double itValue = item.Value;
                if (itValue > disMax) itValue = disMax;
                double sim = simMax - (simMax / disMax * Math.Abs(itValue));
                // insert it into an ordered dictionary
                orderedDictionary.Add(item.Key, sim);
                // update counter
                topK--;
                // check if it is time to break
                if (topK == 0) break;
            }
            // return the value of topK sorted values
            return orderedDictionary;
        }

        internal class Ptr : OpenCvSharp.Ptr
        {
            public Ptr(IntPtr ptr) : base(ptr)
            {
            }

            public override IntPtr Get()
            {
                NativeMethods.HandleException(
                    NativeMethods.face_Ptr_FisherFaceRecognizer_get(ptr, out var ret));
                GC.KeepAlive(this);
                return ret;
            }

            protected override void DisposeUnmanaged()
            {
                NativeMethods.HandleException(
                    NativeMethods.face_Ptr_FisherFaceRecognizer_delete(ptr));
                base.DisposeUnmanaged();
            }
        }
    }
}