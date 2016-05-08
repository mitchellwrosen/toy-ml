{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | Shim between @hmatrix@ and @linear@.

module ML.Internal.Shim
  ( pinv
  ) where

import GHC.TypeLits (Nat)

import qualified Data.Vector           as Vector
import qualified Data.Vector.Generic   as Vector.Generic
import qualified Linear.V              as Linear
import qualified Numeric.LinearAlgebra as HMatrix

-- type LMatrix (n :: Nat) (m :: Nat) = Linear.V n (Linear.V n Double)
-- type HMatrix = HMatrix.Matrix Double

-- | Convert a @linear@ matrix to an @hmatrix@ matrix. Super inefficient.
linear2hmatrix :: forall (n :: Nat) (m :: Nat). Linear.V n (Linear.V m Double) -> HMatrix.Matrix Double
linear2hmatrix =
    HMatrix.fromRows
  . Vector.toList
  . fmap (Vector.Generic.convert . Linear.toVector)
  . Linear.toVector

-- | Convert an @hmatrix@ matrix to a @linear@ matrix. Super inefficient.
hmatrix2linear :: forall (n :: Nat) (m :: Nat). HMatrix.Matrix Double -> Linear.V n (Linear.V m Double)
hmatrix2linear =
    Linear.V
  . Vector.fromList
  . map (Linear.V . Vector.Generic.convert)
  . HMatrix.toRows

withHMatrix
  :: forall (n :: Nat) (m :: Nat) (i :: Nat) (j :: Nat).
     (HMatrix.Matrix Double -> HMatrix.Matrix Double)
  -> Linear.V n (Linear.V m Double)
  -> Linear.V i (Linear.V j Double)
withHMatrix f = hmatrix2linear . f . linear2hmatrix

-- | Pseudo-inverse of a matrix.
pinv :: forall (n :: Nat). Linear.V n (Linear.V n Double) -> Linear.V n (Linear.V n Double)
pinv = withHMatrix HMatrix.pinv
