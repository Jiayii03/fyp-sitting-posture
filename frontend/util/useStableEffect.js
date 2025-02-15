import React, { useEffect } from 'react';

export const useStableEffect = (callback, dependencies) => {
    const previousDeps = React.useRef([]);
  
    useEffect(() => {
      const depsChanged = dependencies.some((dep, index) => dep !== previousDeps.current[index]);
  
      if (depsChanged || previousDeps.current.length === 0) {
        callback();
        previousDeps.current = dependencies;
      }
    }, dependencies);
  };