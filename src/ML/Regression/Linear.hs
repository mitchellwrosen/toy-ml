{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators       #-}

module ML.Regression.Linear where

import ML.Internal.Shim (pinv)

import GHC.TypeLits  (KnownNat)
import Linear.Matrix
import Linear.V
import Linear.Vector

hypothesis :: (Functor t, Foldable f, Additive f, Num a) => t (f a) -> f a -> t a
hypothesis = (!*)

-- |
-- @
-- θ = inv(X'X)X'y
-- @
--
-- The normal equation. Calculates parameters @θ@ corresponding to
-- training data @xs@ and labels @ys@.
normal :: (Foldable t, Additive t, KnownNat n) => t (V n Double) -> t Double -> V n Double
normal xs ys = pinv (xs' !*! xs) !*! xs' !* ys
 where
  xs' = transpose xs
